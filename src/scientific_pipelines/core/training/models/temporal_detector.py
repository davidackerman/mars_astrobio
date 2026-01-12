"""
Temporal Object Detector for Backyard Worlds.

Hybrid 2D+3D CNN architecture for multi-label keypoint detection.
Detects movers and dipoles in 4-frame temporal sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Create a new conv layer with a different input channel count."""
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        avg_weight = conv.weight.mean(dim=1, keepdim=True)
        new_weight = avg_weight.repeat(1, in_channels, 1, 1)
        new_conv.weight.copy_(new_weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class TemporalObjectDetector(nn.Module):
    """
    Hybrid 2D+3D detector for multi-label temporal keypoint detection.

    Architecture:
    1. 2D Spatial Encoder: Pre-trained ResNet18 backbone (per frame)
    2. 3D Temporal Fusion: 3D conv layers to capture motion
    3. Multi-Task Detection Head:
       - Classification: Multi-label presence (mover/dipole)
       - Keypoint Heatmap: Gaussian center-point detection (anchor-free)

    Args:
        num_classes: Number of object classes (default: 2 for mover/dipole)
        pretrained: Use pre-trained ResNet18 weights
        freeze_backbone: Freeze ResNet18 to prevent overfitting
        heatmap_size: Output heatmap resolution (H, W)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        heatmap_size: tuple[int, int] = (8, 8),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.heatmap_size = heatmap_size

        # 2D Spatial Feature Extractor (per-frame)
        # Use ResNet18 pre-trained on ImageNet
        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove final FC and avgpool, keep conv features
        # ResNet18 output: (B*T, 512, H/32, W/32) for input size 256x256
        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False

        # 3D Temporal Fusion
        # Input: (B, 512, T, H', W') where T=4 frames
        # Output: (B, 128, 1, H'', W'') where H'' = heatmap_size
        self.temporal_fusion = nn.Sequential(
            # Temporal convolution to capture motion patterns
            nn.Conv3d(512, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # Spatiotemporal convolution
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Global temporal pooling + spatial downsampling to heatmap size
            nn.AdaptiveAvgPool3d((1, *heatmap_size)),
        )

        # Multi-label Classification Head
        # Predicts presence of each class in the sequence
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * heatmap_size[0] * heatmap_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Keypoint Heatmap Head (anchor-free)
        # Predicts per-pixel heatmap for object centers
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output: num_classes heatmaps (one per class)
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) - Batch of temporal sequences
               where T=4 frames, C=3 (RGB), H=W=256

        Returns:
            class_logits: (B, num_classes) - Multi-label classification logits
            heatmaps: (B, num_classes, H', W') - Keypoint heatmaps
        """
        B, T, C, H, W = x.shape

        # Reshape for 2D processing: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Extract spatial features per frame
        spatial_features = self.spatial_encoder(x)  # (B*T, 512, H', W')

        # Reshape back to temporal: (B, T, 512, H', W') → (B, 512, T, H', W')
        _, C_feat, H_feat, W_feat = spatial_features.shape
        spatial_features = spatial_features.view(B, T, C_feat, H_feat, W_feat)
        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)

        # Temporal fusion
        temporal_features = self.temporal_fusion(spatial_features)  # (B, 128, 1, H'', W'')
        temporal_features = temporal_features.squeeze(2)  # (B, 128, H'', W'')

        # Multi-label classification
        class_logits = self.classifier(temporal_features)  # (B, num_classes)

        # Keypoint heatmap detection
        heatmaps = self.heatmap_head(temporal_features)  # (B, num_classes, H'', W'')

        return class_logits, heatmaps

    def decode_predictions(
        self,
        class_logits: torch.Tensor,
        heatmaps: torch.Tensor,
        score_threshold: float = 0.3,
        nms_kernel: int = 3,
    ) -> list[dict]:
        """
        Decode model predictions into detections.

        Args:
            class_logits: (B, num_classes) classification logits
            heatmaps: (B, num_classes, H, W) keypoint heatmaps
            score_threshold: Minimum confidence score for detection
            nms_kernel: Kernel size for non-maximum suppression

        Returns:
            List of detection dicts per batch item:
            [
                {
                    'class_probs': (num_classes,),
                    'detections': [
                        {'class': int, 'center': (x, y), 'score': float},
                        ...
                    ]
                },
                ...
            ]
        """
        batch_size = class_logits.shape[0]
        class_probs = torch.sigmoid(class_logits)  # (B, num_classes)

        results = []

        for b in range(batch_size):
            batch_detections = {
                'class_probs': class_probs[b].cpu(),
                'detections': []
            }

            # Find peaks in each class heatmap
            for class_id in range(self.num_classes):
                heatmap = torch.sigmoid(heatmaps[b, class_id])  # (H, W)

                # Non-maximum suppression to find local peaks
                peaks = self._find_peaks(heatmap, threshold=score_threshold, kernel_size=nms_kernel)

                for y, x, score in peaks:
                    batch_detections['detections'].append({
                        'class': class_id,
                        'center': (x.item(), y.item()),
                        'score': score.item(),
                    })

            results.append(batch_detections)

        return results

    def _find_peaks(
        self,
        heatmap: torch.Tensor,
        threshold: float,
        kernel_size: int = 3,
    ) -> list[tuple]:
        """
        Find local peaks in heatmap using max pooling.

        Args:
            heatmap: (H, W) heatmap tensor
            threshold: Minimum value for peak
            kernel_size: Size of max pooling kernel

        Returns:
            List of (y, x, score) tuples
        """
        # Apply max pooling to find local maxima
        max_pooled = F.max_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze()

        # Peaks are where value equals max-pooled value and exceeds threshold
        peaks_mask = (heatmap == max_pooled) & (heatmap > threshold)

        # Get coordinates
        peak_coords = torch.nonzero(peaks_mask, as_tuple=False)
        peak_scores = heatmap[peaks_mask]

        return [(y, x, score) for (y, x), score in zip(peak_coords, peak_scores)]


class FrameStackObjectDetector(nn.Module):
    """2D CNN detector using 4-frame channel stacking (12-channel input)."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        heatmap_size: tuple[int, int] = (8, 8),
        in_channels: int = 12,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size

        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        if in_channels != 3:
            resnet.conv1 = _adapt_first_conv(resnet.conv1, in_channels)

        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-2])
        if freeze_backbone:
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False

        self.feature_reduce = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(heatmap_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * heatmap_size[0] * heatmap_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        spatial_features = self.spatial_encoder(x)
        reduced = self.feature_reduce(spatial_features)
        pooled = self.pool(reduced)

        class_logits = self.classifier(pooled)
        heatmaps = self.heatmap_head(pooled)
        return class_logits, heatmaps

    def decode_predictions(
        self,
        class_logits: torch.Tensor,
        heatmaps: torch.Tensor,
        score_threshold: float = 0.3,
        nms_kernel: int = 3,
    ) -> list[dict]:
        batch_size = class_logits.shape[0]
        class_probs = torch.sigmoid(class_logits)

        results = []

        for b in range(batch_size):
            batch_detections = {
                'class_probs': class_probs[b].cpu(),
                'detections': []
            }

            for class_id in range(self.num_classes):
                heatmap = torch.sigmoid(heatmaps[b, class_id])
                peaks = self._find_peaks(heatmap, threshold=score_threshold, kernel_size=nms_kernel)

                for y, x, score in peaks:
                    batch_detections['detections'].append({
                        'class': class_id,
                        'center': (x.item(), y.item()),
                        'score': score.item(),
                    })

            results.append(batch_detections)

        return results

    @staticmethod
    def _find_peaks(
        heatmap: torch.Tensor,
        threshold: float,
        kernel_size: int = 3,
    ) -> list[tuple]:
        max_pooled = F.max_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze()

        peaks_mask = (heatmap == max_pooled) & (heatmap > threshold)
        peak_coords = torch.nonzero(peaks_mask, as_tuple=False)
        peak_scores = heatmap[peaks_mask]

        return [(y, x, score) for (y, x), score in zip(peak_coords, peak_scores)]


class DiffStreamObjectDetector(FrameStackObjectDetector):
    """2D CNN detector using frame differences as input (9-channel input)."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        heatmap_size: tuple[int, int] = (8, 8),
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            heatmap_size=heatmap_size,
            in_channels=9,
        )

    def forward(self, x):
        diffs = x[:, 1:] - x[:, :-1]
        B, Tm1, C, H, W = diffs.shape
        diffs = diffs.view(B, Tm1 * C, H, W)

        spatial_features = self.spatial_encoder(diffs)
        reduced = self.feature_reduce(spatial_features)
        pooled = self.pool(reduced)

        class_logits = self.classifier(pooled)
        heatmaps = self.heatmap_head(pooled)
        return class_logits, heatmaps


def test_temporal_detector():
    """Test the TemporalObjectDetector model."""
    print("Testing TemporalObjectDetector...")

    # Create model
    model = TemporalObjectDetector(
        num_classes=2,
        pretrained=False,  # Faster for testing
        freeze_backbone=True,
        heatmap_size=(8, 8)
    )

    # Create dummy input: batch of 2 sequences, 4 frames, 256x256 RGB
    batch_size = 2
    num_frames = 4
    height, width = 256, 256
    x = torch.randn(batch_size, num_frames, height, width, 3)

    print(f"Input shape: {x.shape}")

    # Forward pass
    class_logits, heatmaps = model(x)

    print(f"Class logits shape: {class_logits.shape}")  # (2, 2)
    print(f"Heatmaps shape: {heatmaps.shape}")  # (2, 2, 8, 8)

    # Decode predictions
    predictions = model.decode_predictions(class_logits, heatmaps)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Sample prediction: {predictions[0]}")

    # Test backward pass
    loss = class_logits.sum() + heatmaps.sum()
    loss.backward()

    print("Backward pass successful!")
    print("Test passed!")


if __name__ == "__main__":
    test_temporal_detector()
