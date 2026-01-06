"""
SimCLR (Simple Framework for Contrastive Learning of Visual Representations).

Implements self-supervised learning via contrastive loss on augmented image pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised representation learning.

    Architecture:
    - Encoder: ResNet backbone (pretrained on ImageNet)
    - Projection head: 2-layer MLP that maps features to contrastive space
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
    ):
        """
        Initialize SimCLR model.

        Parameters
        ----------
        backbone : str
            Backbone architecture ('resnet18', 'resnet50', 'resnet101')
        pretrained : bool
            Whether to use ImageNet pretrained weights
        projection_dim : int
            Dimension of projection space (default: 128)
        hidden_dim : int
            Hidden dimension in projection head (default: 2048)
        """
        super().__init__()

        self.backbone_name = backbone
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        # Load backbone
        if backbone == "resnet18":
            self.encoder = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.encoder = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.encoder = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.feature_dim = feature_dim

        # Projection head (2-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        logger.info(
            f"Initialized SimCLR with {backbone} backbone "
            f"(pretrained={pretrained}, projection_dim={projection_dim})"
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through SimCLR model.

        Parameters
        ----------
        x : Tensor
            Input images, shape (batch_size, 3, H, W)
        return_features : bool
            If True, return both features and projections

        Returns
        -------
        projections : Tensor
            Projected features, shape (batch_size, projection_dim)
        features : Tensor, optional
            Encoder features, shape (batch_size, feature_dim)
            Only returned if return_features=True
        """
        # Extract features with encoder
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten: (B, feature_dim)

        # Project to contrastive space
        projections = self.projection_head(features)

        if return_features:
            return projections, features
        else:
            return projections

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without projection (for similarity search).

        Parameters
        ----------
        x : Tensor
            Input images, shape (batch_size, 3, H, W)

        Returns
        -------
        features : Tensor
            Encoder features, shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.encoder(x)
            features = features.view(features.size(0), -1)
        return features

    def get_embedding_dim(self) -> int:
        """Get dimension of feature embeddings."""
        return self.feature_dim


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR.

    Also known as NT-Xent or InfoNCE loss.
    """

    def __init__(self, temperature: float = 0.5, reduction: str = "mean"):
        """
        Initialize NT-Xent loss.

        Parameters
        ----------
        temperature : float
            Temperature parameter τ (default: 0.5)
        reduction : str
            Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.

        Parameters
        ----------
        z_i : Tensor
            Projections from view 1, shape (batch_size, projection_dim)
        z_j : Tensor
            Projections from view 2, shape (batch_size, projection_dim)

        Returns
        -------
        loss : Tensor
            Contrastive loss value
        """
        batch_size = z_i.size(0)

        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate projections: [z_i; z_j]
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)

        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.t())  # (2*batch_size, 2*batch_size)

        # Create masks for positive and negative pairs
        # Positive pairs: (i, i+N) and (i+N, i) where N = batch_size
        # Negative pairs: all other pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # Positive pair indices
        pos_indices = torch.arange(batch_size, device=z.device)

        # Compute loss for each sample
        losses = []
        for i in range(2 * batch_size):
            # Get positive pair index
            if i < batch_size:
                pos_idx = i + batch_size  # View 1 → View 2
            else:
                pos_idx = i - batch_size  # View 2 → View 1

            # Positive similarity
            pos_sim = similarity_matrix[i, pos_idx] / self.temperature

            # Negative similarities (all except self and positive)
            neg_mask = ~mask[i].clone()
            neg_mask[pos_idx] = False
            neg_sim = similarity_matrix[i, neg_mask] / self.temperature

            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # Numerically stable version using log-sum-exp
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
            loss_i = -pos_sim + torch.logsumexp(logits, dim=0)
            losses.append(loss_i)

        # Aggregate losses
        loss = torch.stack(losses)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SimCLRWithLinearProbe(nn.Module):
    """
    SimCLR encoder with a linear probe for evaluation.

    Used to evaluate the quality of learned representations by training
    a linear classifier on frozen features.
    """

    def __init__(
        self,
        simclr_model: SimCLR,
        num_classes: int = 5,
        freeze_encoder: bool = True,
    ):
        """
        Initialize SimCLR with linear probe.

        Parameters
        ----------
        simclr_model : SimCLR
            Pretrained SimCLR model
        num_classes : int
            Number of classification classes (default: 5)
        freeze_encoder : bool
            Whether to freeze encoder weights (default: True)
        """
        super().__init__()

        self.encoder = simclr_model.encoder
        self.feature_dim = simclr_model.feature_dim

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Linear classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        logger.info(
            f"Initialized linear probe (num_classes={num_classes}, "
            f"freeze_encoder={freeze_encoder})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.

        Parameters
        ----------
        x : Tensor
            Input images, shape (batch_size, 3, H, W)

        Returns
        -------
        logits : Tensor
            Classification logits, shape (batch_size, num_classes)
        """
        # Extract features
        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        # Classify
        logits = self.classifier(features)
        return logits


def create_simclr_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    projection_dim: int = 128,
    hidden_dim: int = 2048,
) -> SimCLR:
    """
    Factory function to create SimCLR model.

    Parameters
    ----------
    backbone : str
        Backbone architecture
    pretrained : bool
        Use ImageNet pretrained weights
    projection_dim : int
        Projection space dimension
    hidden_dim : int
        Hidden dimension in projection head

    Returns
    -------
    model : SimCLR
        Initialized SimCLR model
    """
    model = SimCLR(
        backbone=backbone,
        pretrained=pretrained,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
    )
    return model


if __name__ == "__main__":
    # Test SimCLR model
    print("Testing SimCLR model...")

    # Create model
    model = create_simclr_model(backbone="resnet50", pretrained=True)
    print(f"Model created: {model.backbone_name}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Projection dimension: {model.projection_dim}")

    # Test forward pass
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)

    # Get projections
    z1 = model(x1)
    z2 = model(x2)
    print(f"\nProjection shape: {z1.shape}")

    # Get features
    features = model.get_features(x1)
    print(f"Feature shape: {features.shape}")

    # Test loss
    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z1, z2)
    print(f"\nNT-Xent loss: {loss.item():.4f}")

    # Test linear probe
    probe = SimCLRWithLinearProbe(model, num_classes=5, freeze_encoder=True)
    logits = probe(x1)
    print(f"\nLinear probe output shape: {logits.shape}")

    print("\nAll tests passed!")
