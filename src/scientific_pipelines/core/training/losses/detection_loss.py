"""
Detection Loss for temporal object detection.

Multi-task loss combining:
1. Classification loss (multi-label binary cross-entropy)
2. Keypoint heatmap loss (focal loss for class imbalance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Multi-task loss for temporal keypoint detection.

    Components:
    1. Classification loss: Binary cross-entropy for multi-label
    2. Heatmap loss: Focal loss for center point detection

    Args:
        alpha_cls: Weight for classification loss
        alpha_heatmap: Weight for heatmap loss
        focal_alpha: Focal loss alpha parameter (class balance)
        focal_gamma: Focal loss gamma parameter (focus on hard examples)
    """

    def __init__(
        self,
        alpha_cls: float = 1.0,
        alpha_heatmap: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.alpha_heatmap = alpha_heatmap
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.bce_loss = nn.BCEWithLogitsLoss()

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for imbalanced heatmap detection.

        Addresses class imbalance (most pixels are background).

        Args:
            pred: (B, C, H, W) predicted heatmap logits
            target: (B, C, H, W) target heatmap (0 or 1)

        Returns:
            Scalar focal loss
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Compute probability of the true class
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.focal_gamma

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Weighted focal loss
        loss = self.focal_alpha * focal_weight * bce

        return loss.mean()

    def forward(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss.

        Args:
            predictions: (class_logits, heatmaps)
                - class_logits: (B, num_classes) classification logits
                - heatmaps: (B, num_classes, H, W) heatmap logits
            targets: (class_labels, heatmap_targets)
                - class_labels: (B, num_classes) binary labels
                - heatmap_targets: (B, num_classes, H, W) Gaussian heatmaps

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses for logging
        """
        class_logits, heatmaps = predictions
        class_labels, heatmap_targets = targets

        # 1. Classification loss (multi-label)
        loss_cls = self.bce_loss(class_logits, class_labels)

        # 2. Heatmap loss (focal loss for center detection)
        loss_heatmap = self.focal_loss(heatmaps, heatmap_targets)

        # Total weighted loss
        total_loss = (
            self.alpha_cls * loss_cls +
            self.alpha_heatmap * loss_heatmap
        )

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_cls': loss_cls.item(),
            'loss_heatmap': loss_heatmap.item(),
        }

        return total_loss, loss_dict


def test_detection_loss():
    """Test the DetectionLoss."""
    print("Testing DetectionLoss...")

    # Create loss function
    loss_fn = DetectionLoss(
        alpha_cls=1.0,
        alpha_heatmap=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )

    # Create dummy predictions
    batch_size = 2
    num_classes = 2
    heatmap_size = (8, 8)

    class_logits = torch.randn(batch_size, num_classes)
    heatmaps = torch.randn(batch_size, num_classes, *heatmap_size)

    # Create dummy targets
    class_labels = torch.zeros(batch_size, num_classes)
    class_labels[0, 0] = 1  # First sample has mover
    class_labels[1, 1] = 1  # Second sample has dipole

    heatmap_targets = torch.zeros(batch_size, num_classes, *heatmap_size)
    # Add Gaussian blob at (4, 4) for first sample, mover class
    heatmap_targets[0, 0, 3:6, 3:6] = 0.5
    heatmap_targets[0, 0, 4, 4] = 1.0

    print(f"Class logits shape: {class_logits.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Class labels shape: {class_labels.shape}")
    print(f"Heatmap targets shape: {heatmap_targets.shape}")

    # Compute loss
    predictions = (class_logits, heatmaps)
    targets = (class_labels, heatmap_targets)

    total_loss, loss_dict = loss_fn(predictions, targets)

    print(f"\nLoss values:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Classification: {loss_dict['loss_cls']:.4f}")
    print(f"  Heatmap: {loss_dict['loss_heatmap']:.4f}")

    # Test backward pass
    total_loss.backward()

    print("\nBackward pass successful!")
    print("Test passed!")


if __name__ == "__main__":
    test_detection_loss()
