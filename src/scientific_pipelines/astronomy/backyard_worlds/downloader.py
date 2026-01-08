"""Download Backyard Worlds flipbook subjects from Zooniverse."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BackyardWorldsDownloader:
    """
    Download Backyard Worlds flipbook subjects from Zooniverse Panoptes API.

    Backyard Worlds: Planet 9 is a citizen science project that uses flipbooks
    (sequences of WISE infrared images) to search for brown dwarfs and planets.

    Args:
        output_dir: Directory to save downloaded subjects
        username: Optional Panoptes username for higher API rate limits
        password: Optional Panoptes password
        max_retries: Maximum number of download retries (default: 3)
        retry_delay: Delay between retries in seconds (default: 5)
    """

    BACKYARD_WORLDS_PROJECT_ID = 1901  # Backyard Worlds: Planet 9

    def __init__(
        self,
        output_dir: Path,
        username: Optional[str] = None,
        password: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Try to use Panoptes client if available
        self.use_panoptes = False
        try:
            from panoptes_client import Panoptes, Project

            # Login if credentials provided
            if username and password:
                logger.info(f"Logging into Panoptes as {username}")
                Panoptes.connect(username=username, password=password)
            else:
                logger.info("Connecting to Panoptes anonymously")
                Panoptes.connect()

            self.project = Project.find(self.BACKYARD_WORLDS_PROJECT_ID)
            self.use_panoptes = True
            logger.info(f"Connected to project: {self.project.display_name}")

        except ImportError:
            logger.warning(
                "panoptes-client not available. "
                "Install with: pip install panoptes-client"
            )
            logger.info("Will use direct API access instead")
            self.use_panoptes = False

    def download_subjects(
        self,
        limit: int = 1000,
        subject_set_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Download subject flipbooks.

        Args:
            limit: Maximum number of subjects to download
            subject_set_id: Optional specific subject set ID to download from

        Returns:
            List of subject metadata dictionaries with local frame paths
        """
        if self.use_panoptes:
            return self._download_with_panoptes(limit, subject_set_id)
        else:
            return self._download_with_api(limit, subject_set_id)

    def _download_with_panoptes(
        self,
        limit: int,
        subject_set_id: Optional[int],
    ) -> List[Dict]:
        """Download subjects using Panoptes Python client."""
        from panoptes_client import SubjectSet

        logger.info(f"Downloading up to {limit} subjects using Panoptes client")

        subjects_metadata = []

        # Get subjects from project or specific subject set
        if subject_set_id:
            subject_set = SubjectSet.find(subject_set_id)
            subjects_iter = subject_set.subjects
            logger.info(f"Downloading from subject set {subject_set_id}")
        else:
            subjects_iter = self.project.subjects
            logger.info("Downloading from all project subjects")

        # Iterate through subjects
        for i, subject in enumerate(tqdm(subjects_iter, desc="Downloading subjects", total=limit)):
            if i >= limit:
                break

            # Create subject directory
            subject_dir = self.output_dir / f"subject_{subject.id}"
            subject_dir.mkdir(exist_ok=True, parents=True)

            # Download each frame in the flipbook
            frame_paths = []
            for location_idx, (location_key, location_url) in enumerate(
                subject.locations.items()
            ):
                # Determine file extension from URL
                parsed_url = urlparse(location_url)
                ext = Path(parsed_url.path).suffix or '.fits'

                # Download frame
                frame_filename = f"frame_{location_idx:02d}{ext}"
                frame_path = subject_dir / frame_filename

                if not frame_path.exists():
                    self._download_file(location_url, frame_path)

                frame_paths.append(str(frame_path))

            # Store metadata
            subjects_metadata.append(
                {
                    'subject_id': subject.id,
                    'frame_paths': frame_paths,
                    'n_frames': len(frame_paths),
                    'metadata': subject.metadata,
                    'ra': subject.metadata.get('!RA'),
                    'dec': subject.metadata.get('!Dec'),
                }
            )

        logger.info(f"Downloaded {len(subjects_metadata)} subjects")
        return subjects_metadata

    def _download_with_api(
        self,
        limit: int,
        subject_set_id: Optional[int],
    ) -> List[Dict]:
        """Download subjects using direct API calls (fallback)."""
        logger.info(f"Downloading up to {limit} subjects using direct API")

        # Build API URL
        base_url = "https://www.zooniverse.org/api/subjects"
        params = {
            'project_id': self.BACKYARD_WORLDS_PROJECT_ID,
            'page_size': min(100, limit),  # Max 100 per page
        }

        if subject_set_id:
            params['subject_set_id'] = subject_set_id

        subjects_metadata = []
        page = 1

        with requests.Session() as session:
            while len(subjects_metadata) < limit:
                params['page'] = page

                # Make API request
                response = session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                subjects = data.get('subjects', [])
                if not subjects:
                    break  # No more subjects

                # Process each subject
                for subject_data in subjects:
                    if len(subjects_metadata) >= limit:
                        break

                    subject_id = subject_data['id']
                    subject_dir = self.output_dir / f"subject_{subject_id}"
                    subject_dir.mkdir(exist_ok=True, parents=True)

                    # Download frames
                    frame_paths = []
                    locations = subject_data.get('locations', [])

                    for location_idx, location in enumerate(locations):
                        # Get URL (locations can be dict or list)
                        if isinstance(location, dict):
                            location_url = list(location.values())[0]
                        else:
                            location_url = location

                        # Determine extension
                        parsed_url = urlparse(location_url)
                        ext = Path(parsed_url.path).suffix or '.fits'

                        frame_filename = f"frame_{location_idx:02d}{ext}"
                        frame_path = subject_dir / frame_filename

                        if not frame_path.exists():
                            self._download_file(location_url, frame_path)

                        frame_paths.append(str(frame_path))

                    # Store metadata
                    metadata = subject_data.get('metadata', {})
                    subjects_metadata.append(
                        {
                            'subject_id': subject_id,
                            'frame_paths': frame_paths,
                            'n_frames': len(frame_paths),
                            'metadata': metadata,
                            'ra': metadata.get('!RA'),
                            'dec': metadata.get('!Dec'),
                        }
                    )

                page += 1
                time.sleep(0.5)  # Rate limiting

        logger.info(f"Downloaded {len(subjects_metadata)} subjects")
        return subjects_metadata

    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a single file with retry logic.

        Args:
            url: URL to download from
            output_path: Path to save file

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()

                # Write to file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return True

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Download failed after {self.max_retries} attempts: {e}")
                    return False

        return False
