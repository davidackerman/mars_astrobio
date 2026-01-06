"""
PyTorch Dataset classes for Mars rover imagery.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class WATSONTextureDataset(Dataset):
    """
    PyTorch Dataset for WATSON micro-imager texture classification.

    Loads WATSON images and corresponding texture labels for training
    deep learning models to recognize biosignature-proxy textures.
    """

    # Texture class definitions
    TEXTURE_CLASSES = [
        "laminations",  # Fine-scale layering
        "porosity",     # Micro-porous textures
        "halos",        # Iron oxide halos
        "alteration",   # Chemical alteration zones
        "normal",       # Background rock textures
    ]

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        annotation_file: Optional[Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize WATSON texture dataset.

        Parameters
        ----------
        root : Path
            Root directory containing processed data.
            Expected structure: root/split/class_name/images
        split : str
            Dataset split: "train", "val", or "test"
        transform : callable, optional
            Transform to apply to images
        annotation_file : Path, optional
            Path to JSON annotation file with labels
        class_to_idx : dict, optional
            Mapping of class names to indices. If None, uses alphabetical order.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Set up class mapping
        if class_to_idx is None:
            self.class_to_idx = {
                cls: idx for idx, cls in enumerate(sorted(self.TEXTURE_CLASSES))
            }
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Load samples
        if annotation_file and annotation_file.exists():
            self.samples = self._load_from_annotations(annotation_file)
        else:
            self.samples = self._load_from_directory()

        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"({self.num_classes} classes)"
        )

    def _load_from_directory(self) -> List[Tuple[Path, int]]:
        """
        Load samples from directory structure.

        Expected structure:
        root/
            train/
                laminations/
                    image1.jpg
                    image2.jpg
                porosity/
                    image3.jpg
            val/
                ...

        Returns
        -------
        list of (Path, int)
            List of (image_path, class_idx) tuples
        """
        samples = []
        split_dir = self.root / self.split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return samples

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name

            if not class_dir.exists():
                logger.debug(f"Class directory not found: {class_dir}")
                continue

            # Find all image files
            image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".img"}
            for ext in image_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    samples.append((img_path, class_idx))

        return samples

    def _load_from_annotations(self, annotation_file: Path) -> List[Tuple[Path, int]]:
        """
        Load samples from annotation JSON file.

        Annotation format:
        {
            "images": [
                {
                    "path": "path/to/image.jpg",
                    "label": "laminations",
                    "confidence": 0.9
                },
                ...
            ]
        }

        Parameters
        ----------
        annotation_file : Path
            Path to JSON annotation file

        Returns
        -------
        list of (Path, int)
            List of (image_path, class_idx) tuples
        """
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        samples = []
        for item in annotations.get("images", []):
            img_path = Path(item["path"])
            label = item.get("label")

            if label not in self.class_to_idx:
                logger.warning(f"Unknown label '{label}' for {img_path}")
                continue

            class_idx = self.class_to_idx[label]

            # Make path relative to root if needed
            if not img_path.is_absolute():
                img_path = self.root / img_path

            samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        tuple of (Tensor, int)
            (image, label) pair
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes in dataset.

        Returns
        -------
        dict
            Mapping of class name to sample count
        """
        distribution = {cls: 0 for cls in self.class_to_idx.keys()}

        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1

        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.

        Uses inverse frequency weighting.

        Returns
        -------
        Tensor
            Weight for each class, shape (num_classes,)
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())

        weights = torch.zeros(self.num_classes)
        for class_name, count in distribution.items():
            if count > 0:
                class_idx = self.class_to_idx[class_name]
                weights[class_idx] = total / (self.num_classes * count)

        return weights


class UnsupervisedDataset(Dataset):
    """
    Unsupervised dataset for contrastive learning (SimCLR).

    Loads images without labels and applies two different augmentations
    to create positive pairs for contrastive learning.
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.tif', '.tiff'),
    ):
        """
        Initialize unsupervised dataset.

        Parameters
        ----------
        root : Path
            Root directory containing images (recursively searched)
        transform : callable, optional
            Transform to apply twice to each image for contrastive pairs
        extensions : tuple
            Image file extensions to include
        """
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions

        # Recursively find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.root.rglob(f"*{ext}")))

        # Sort for reproducibility
        self.image_paths = sorted(self.image_paths)

        logger.info(f"Found {len(self.image_paths)} images in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get two augmented views of the same image.

        Parameters
        ----------
        idx : int
            Image index

        Returns
        -------
        tuple of (Tensor, Tensor)
            Two different augmentations of the same image (positive pair)
        """
        img_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return black image as fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply transform twice to get two views
        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
        else:
            # Default: just convert to tensor
            from torchvision import transforms as T
            to_tensor = T.ToTensor()
            view1 = to_tensor(image)
            view2 = to_tensor(image)

        return view1, view2

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for an image."""
        return self.image_paths[idx]


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal Mars rover data (WATSON + SHERLOC + PIXL).

    Future implementation for integrating imaging, spectroscopy, and
    elemental composition data.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        watson_transform: Optional[Callable] = None,
        sherloc_transform: Optional[Callable] = None,
        pixl_transform: Optional[Callable] = None,
    ):
        """
        Initialize multi-modal dataset.

        Parameters
        ----------
        root : Path
            Root directory for multi-modal data
        split : str
            Dataset split
        watson_transform : callable, optional
            Transform for WATSON images
        sherloc_transform : callable, optional
            Transform for SHERLOC spectra
        pixl_transform : callable, optional
            Transform for PIXL elemental maps
        """
        self.root = Path(root)
        self.split = split
        self.watson_transform = watson_transform
        self.sherloc_transform = sherloc_transform
        self.pixl_transform = pixl_transform

        # TODO: Implement multi-modal data loading
        raise NotImplementedError(
            "Multi-modal dataset will be implemented in Phase 6"
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
