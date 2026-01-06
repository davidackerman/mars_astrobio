"""
WATSON (Wide Angle Topographic Sensor for Operations and eNgineering) downloader.

Downloads WATSON micro-imager images from NASA's Mars 2020 mission.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from tqdm import tqdm

from mars_biosig.data.pds_client import PDSClient

logger = logging.getLogger(__name__)


class WATSONDownloader:
    """
    Downloader for WATSON micro-imager images.

    WATSON is a high-resolution color camera on the end of Perseverance's
    robotic arm, used for close-up imaging of rock textures and features.
    """

    def __init__(
        self,
        output_dir: Path = Path("data/raw/watson"),
        api_key: Optional[str] = None,
    ):
        """
        Initialize WATSON downloader.

        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded images
        api_key : str, optional
            NASA API key (or set NASA_API_KEY environment variable)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = PDSClient(api_key=api_key)
        self.manifest_path = self.output_dir / "manifest.json"

        # Load existing manifest
        self.manifest = self._load_manifest()

        logger.info(f"Initialized WATSON downloader, output: {self.output_dir}")

    def _load_manifest(self) -> Dict:
        """Load download manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {
            "downloaded_sols": [],
            "total_images": 0,
            "last_updated": None,
        }

    def _save_manifest(self):
        """Save download manifest to disk."""
        self.manifest["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        logger.debug(f"Saved manifest: {self.manifest_path}")

    def _is_calibration_target(self, photo: Dict) -> bool:
        """
        Check if photo is a calibration target.

        Parameters
        ----------
        photo : dict
            Photo metadata from API

        Returns
        -------
        bool
            True if photo is a calibration target
        """
        # Calibration targets often have specific patterns in image name
        img_src = photo.get("img_src", "").lower()

        # Common calibration target indicators
        cal_keywords = [
            "cal",
            "calibration",
            "target",
            "flat_field",
            "dark",
        ]

        return any(keyword in img_src for keyword in cal_keywords)

    def _get_sol_dir(self, sol: int) -> Path:
        """Get directory path for a sol."""
        return self.output_dir / f"sol_{sol:04d}"

    def download_sol(
        self,
        sol: int,
        skip_calibration: bool = True,
        overwrite: bool = False,
        data_type: str = "edr",
    ) -> List[Path]:
        """
        Download all WATSON images for a specific sol from PDS archive.

        Parameters
        ----------
        sol : int
            Martian sol number
        skip_calibration : bool
            Whether to skip calibration target images (default: True)
        overwrite : bool
            Whether to overwrite existing images (default: False)
        data_type : str
            Data product type: 'edr' (raw), 'rdr' (reduced), 'fdr' (formatted)

        Returns
        -------
        list of Path
            Paths to downloaded IMG files
        """
        logger.info(f"Downloading WATSON IMG files for sol {sol} ({data_type})")

        # List available images from PDS archive
        images = self.client.list_watson_images(sol=sol, data_type=data_type)

        if not images:
            logger.warning(f"No WATSON images found for sol {sol}")
            return []

        logger.info(f"Found {len(images)} IMG files for sol {sol}")

        # Create sol directory
        sol_dir = self._get_sol_dir(sol)
        sol_dir.mkdir(parents=True, exist_ok=True)

        # Download each image
        downloaded_files = []
        for image_info in tqdm(images, desc=f"Sol {sol}", unit="img"):
            try:
                img_path, label_path = self.client.download_watson_image(
                    image_info,
                    sol_dir,
                    download_label=True,
                    overwrite=overwrite,
                )
                if img_path:
                    downloaded_files.append(img_path)
            except Exception as e:
                logger.error(f"Failed to download {image_info['filename']}: {e}")
                continue

        # Update manifest
        if sol not in self.manifest["downloaded_sols"]:
            self.manifest["downloaded_sols"].append(sol)
            self.manifest["downloaded_sols"].sort()

        self.manifest["total_images"] += len(downloaded_files)
        self._save_manifest()

        logger.info(f"Downloaded {len(downloaded_files)} IMG files for sol {sol}")
        return downloaded_files

    def _download_photo(
        self,
        photo: Dict,
        sol_dir: Path,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Download a single photo and its metadata.

        Parameters
        ----------
        photo : dict
            Photo metadata from API
        sol_dir : Path
            Directory to save photo
        overwrite : bool
            Whether to overwrite existing file

        Returns
        -------
        Path or None
            Path to downloaded file, or None if skipped
        """
        photo_id = photo.get("id")
        img_url = photo.get("img_src")

        if not img_url:
            logger.warning(f"No image URL for photo {photo_id}")
            return None

        # Construct filename from URL
        filename = Path(img_url).name
        img_path = sol_dir / filename

        # Check if already exists
        if img_path.exists() and not overwrite:
            logger.debug(f"Skipping existing file: {filename}")
            return img_path

        # Download image
        try:
            self.client.download_file(
                img_url,
                img_path,
                overwrite=overwrite,
                show_progress=False,  # Using tqdm instead
            )
        except Exception as e:
            logger.error(f"Failed to download {img_url}: {e}")
            return None

        # Save metadata as JSON
        metadata_path = img_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(photo, f, indent=2)

        return img_path

    def download_sol_range(
        self,
        start_sol: int,
        end_sol: int,
        skip_calibration: bool = True,
        overwrite: bool = False,
        data_type: str = "edr",
    ) -> Dict[int, List[Path]]:
        """
        Download WATSON images for a range of sols from PDS archive.

        Parameters
        ----------
        start_sol : int
            Starting sol (inclusive)
        end_sol : int
            Ending sol (inclusive)
        skip_calibration : bool
            Whether to skip calibration targets (currently unused for IMG files)
        overwrite : bool
            Whether to overwrite existing files
        data_type : str
            Data product type: 'edr' (raw), 'rdr' (reduced), 'fdr' (formatted)

        Returns
        -------
        dict
            Mapping of sol -> list of downloaded file paths
        """
        logger.info(f"Downloading sols {start_sol} to {end_sol}")

        # Get list of available sols from PDS
        available_sols = self.client.list_watson_sols()
        target_sols = [s for s in available_sols if start_sol <= s <= end_sol]

        logger.info(f"Found {len(target_sols)} sols with data in range {start_sol}-{end_sol}")

        results = {}
        for sol in target_sols:
            try:
                files = self.download_sol(
                    sol=sol,
                    skip_calibration=skip_calibration,
                    overwrite=overwrite,
                    data_type=data_type,
                )
                results[sol] = files
            except Exception as e:
                logger.error(f"Failed to download sol {sol}: {e}")
                results[sol] = []
                continue

        total_images = sum(len(files) for files in results.values())
        logger.info(
            f"Downloaded {total_images} IMG files across "
            f"{len([s for s in results.values() if s])} sols"
        )

        return results

    def get_downloaded_sols(self) -> List[int]:
        """Get list of sols that have been downloaded."""
        return sorted(self.manifest["downloaded_sols"])

    def get_statistics(self) -> Dict:
        """
        Get download statistics.

        Returns
        -------
        dict
            Statistics including total images, sols, date ranges
        """
        downloaded_sols = self.get_downloaded_sols()

        return {
            "total_images": self.manifest["total_images"],
            "total_sols": len(downloaded_sols),
            "sol_range": (
                (min(downloaded_sols), max(downloaded_sols))
                if downloaded_sols
                else None
            ),
            "last_updated": self.manifest["last_updated"],
            "output_dir": str(self.output_dir),
        }

    def close(self):
        """Close the PDS client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
