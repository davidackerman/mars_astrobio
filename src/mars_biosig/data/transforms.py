"""
Data augmentation and preprocessing transforms for Mars rover imagery.

Includes Mars-specific augmentations that preserve texture information.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from typing import List, Optional
import random


class MarsImageTransform:
    """
    Base class for Mars image transforms.

    Provides common transformations suitable for Mars rover imagery,
    focusing on preserving texture information critical for biosignature detection.
    """

    @staticmethod
    def get_train_transform(
        image_size: int = 224,
        normalize: bool = True,
    ) -> T.Compose:
        """
        Get training data augmentation pipeline.

        Parameters
        ----------
        image_size : int
            Target image size (default: 224 for ResNet compatibility)
        normalize : bool
            Whether to normalize with ImageNet stats (default: True)

        Returns
        -------
        Compose
            Composed transform pipeline
        """
        transforms = [
            # Resize and random crop for scale variation
            T.Resize(int(image_size * 1.15)),
            T.RandomCrop(image_size),

            # Flips - preserve texture symmetry
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),

            # Random rotation in 90-degree increments (preserve texture orientation)
            T.RandomApply([T.RandomRotation([0, 0, 90, 90, 180, 180, 270, 270])], p=0.5),

            # Color jitter for lighting variations (Mars has different lighting conditions)
            T.ColorJitter(
                brightness=0.2,  # Dust, shadows
                contrast=0.2,    # Different exposure settings
                saturation=0.1,  # Atmospheric conditions
                hue=0.05,        # Slight color cast variations
            ),

            # Random grayscale to make model robust to color variations
            T.RandomGrayscale(p=0.1),

            # Gaussian blur for defocus simulation
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),

            # Convert to tensor
            T.ToTensor(),
        ]

        # Add normalization if requested
        if normalize:
            # Use ImageNet statistics (common practice for transfer learning)
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return T.Compose(transforms)

    @staticmethod
    def get_val_transform(
        image_size: int = 224,
        normalize: bool = True,
    ) -> T.Compose:
        """
        Get validation/test data preprocessing pipeline.

        Parameters
        ----------
        image_size : int
            Target image size (default: 224)
        normalize : bool
            Whether to normalize with ImageNet stats (default: True)

        Returns
        -------
        Compose
            Composed transform pipeline
        """
        transforms = [
            # Resize and center crop
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),

            # Convert to tensor
            T.ToTensor(),
        ]

        # Add normalization if requested
        if normalize:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return T.Compose(transforms)

    @staticmethod
    def get_inference_transform(
        image_size: int = 224,
        normalize: bool = True,
    ) -> T.Compose:
        """
        Get inference preprocessing pipeline.

        Same as validation transform - no augmentation.

        Parameters
        ----------
        image_size : int
            Target image size
        normalize : bool
            Whether to normalize

        Returns
        -------
        Compose
            Composed transform pipeline
        """
        return MarsImageTransform.get_val_transform(image_size, normalize)


class TexturePreservingAugmentation:
    """
    Advanced augmentation that preserves texture patterns.

    For biosignature detection, we need to preserve fine-scale textures
    while still providing data augmentation.
    """

    def __init__(
        self,
        image_size: int = 224,
        crop_scale: tuple = (0.8, 1.0),
        rotation_degrees: int = 15,
    ):
        """
        Initialize texture-preserving augmentation.

        Parameters
        ----------
        image_size : int
            Output image size
        crop_scale : tuple
            Range of crop scale (min, max)
        rotation_degrees : int
            Max rotation angle (degrees)
        """
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.rotation_degrees = rotation_degrees

    def __call__(self, img):
        """
        Apply texture-preserving augmentation.

        Parameters
        ----------
        img : PIL.Image
            Input image

        Returns
        -------
        Tensor
            Augmented image tensor
        """
        # Random resized crop that preserves texture details
        i, j, h, w = T.RandomResizedCrop.get_params(
            img,
            scale=self.crop_scale,
            ratio=(0.9, 1.1),  # Keep aspect ratio close to square
        )
        img = TF.resized_crop(img, i, j, h, w, (self.image_size, self.image_size))

        # Small rotation to preserve texture orientation
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = TF.rotate(img, angle)

        # Random flips
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)

        # Gentle color jitter
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))

        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return img


class MarsNormalization:
    """
    Mars-specific normalization.

    Alternative to ImageNet normalization, using statistics
    computed from Mars rover imagery.
    """

    # TODO: Compute actual statistics from Mars rover data
    # These are placeholder values
    MARS_MEAN = [0.45, 0.35, 0.25]  # Reddish tint on Mars
    MARS_STD = [0.20, 0.20, 0.20]

    def __init__(self, use_mars_stats: bool = True):
        """
        Initialize Mars normalization.

        Parameters
        ----------
        use_mars_stats : bool
            If True, use Mars statistics. If False, use ImageNet statistics.
        """
        if use_mars_stats:
            self.mean = self.MARS_MEAN
            self.std = self.MARS_STD
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor.

        Parameters
        ----------
        tensor : Tensor
            Input image tensor (C, H, W)

        Returns
        -------
        Tensor
            Normalized tensor
        """
        return TF.normalize(tensor, mean=self.mean, std=self.std)


def get_transform_config(
    mode: str = "train",
    image_size: int = 224,
    use_texture_preserving: bool = False,
) -> T.Compose:
    """
    Get transform configuration by mode.

    Parameters
    ----------
    mode : str
        One of "train", "val", "test", "inference"
    image_size : int
        Target image size
    use_texture_preserving : bool
        Use advanced texture-preserving augmentation for training

    Returns
    -------
    Compose
        Transform pipeline
    """
    if mode == "train":
        if use_texture_preserving:
            return TexturePreservingAugmentation(image_size=image_size)
        else:
            return MarsImageTransform.get_train_transform(image_size=image_size)
    elif mode in ["val", "test", "inference"]:
        return MarsImageTransform.get_val_transform(image_size=image_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")
