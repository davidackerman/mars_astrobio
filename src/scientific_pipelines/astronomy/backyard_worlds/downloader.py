"""Download Backyard Worlds flipbook subjects from Zooniverse."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime

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

    BACKYARD_WORLDS_PROJECT_ID = 2416  # Backyard Worlds: Planet 9

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

        # Use direct API access (more reliable than Panoptes client)
        self.use_panoptes = False
        self.username = username
        self.password = password

        logger.info("Using direct Zooniverse API access")
        if username and password:
            logger.info(f"Will authenticate as {username}")
        else:
            logger.info("Using anonymous access (may have lower rate limits)")

    def download_subjects(
        self,
        limit: int = 1000,
        subject_set_id: Optional[int] = None,
        retired: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Download subject flipbooks.

        Args:
            limit: Maximum number of subjects to download
            subject_set_id: Optional specific subject set ID to download from
            retired: Optional filter - True for retired (examined), False for active (unexamined), None for all

        Returns:
            List of subject metadata dictionaries with local frame paths
        """
        return self._download_with_panoptes(limit, subject_set_id, retired)

    def _download_with_panoptes(
        self,
        limit: int,
        subject_set_id: Optional[int],
        retired: Optional[bool],
    ) -> List[Dict]:
        """Download subjects using Panoptes Python client."""
        from panoptes_client import Panoptes, Project, SubjectSet, Subject

        filter_msg = ""
        if retired is True:
            filter_msg = " (retired/examined only)"
        elif retired is False:
            filter_msg = " (active/unexamined only)"

        logger.info(f"Downloading up to {limit} subjects{filter_msg} using Panoptes client")

        # Connect to Panoptes
        if self.username and self.password:
            Panoptes.connect(username=self.username, password=self.password)
        else:
            Panoptes.connect()

        # Get project
        project = Project.find(self.BACKYARD_WORLDS_PROJECT_ID)

        subjects_metadata = []

        # Get subjects based on retirement status
        if retired is not None:
            # Use direct Subject query with retirement filter
            logger.info(f"Querying subjects with retired={retired}")
            subjects_iter = Subject.where(project_id=self.BACKYARD_WORLDS_PROJECT_ID, retired=retired)
            logger.info(f"Downloading {'retired' if retired else 'active'} subjects from project")
        elif subject_set_id:
            subject_set = SubjectSet.find(subject_set_id)
            subjects_iter = subject_set.subjects
            logger.info(f"Downloading from subject set {subject_set_id}")
        else:
            # Get first subject set (they're already loaded in project.links.subject_sets)
            subject_sets = project.links.subject_sets
            if not subject_sets:
                raise ValueError("No subject sets found for project")

            subject_set = subject_sets[0]
            subjects_iter = subject_set.subjects
            logger.info(f"Downloading from subject set: {subject_set.display_name}")

        # Iterate through subjects
        downloaded_count = 0
        for subject in tqdm(subjects_iter, desc="Downloading subjects", total=limit):
            if downloaded_count >= limit:
                break

            # Create subject directory
            subject_dir = self.output_dir / f"subject_{subject.id}"
            subject_dir.mkdir(exist_ok=True, parents=True)

            # Download each frame in the flipbook
            # Locations is a list of dicts like [{'image/jpeg': 'url1'}, {'image/jpeg': 'url2'}]
            frame_paths = []
            for location_idx, location_dict in enumerate(subject.locations):
                # Extract URL from the location dictionary
                mime_type = list(location_dict.keys())[0]
                location_url = location_dict[mime_type]

                # Determine file extension from MIME type or URL
                if 'jpeg' in mime_type or 'jpg' in mime_type:
                    ext = '.jpg'
                elif 'png' in mime_type:
                    ext = '.png'
                elif 'fits' in mime_type:
                    ext = '.fits'
                else:
                    # Fallback to URL extension
                    parsed_url = urlparse(location_url)
                    ext = Path(parsed_url.path).suffix or '.jpg'

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

            downloaded_count += 1

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

    def download_classifications(
        self,
        subject_ids: Optional[List[int]] = None,
        workflow_id: Optional[int] = None,
        limit: Optional[int] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> List[Dict]:
        """
        Download classification data for subjects.

        Note: The Panoptes API may not return all classifications through the standard
        query interface. For complete data, you may need to request a data export from
        the Zooniverse project page or use the project builder's data export feature.

        Args:
            subject_ids: Optional list of specific subject IDs to get classifications for
            workflow_id: Optional workflow ID to filter classifications
            limit: Optional maximum number of classifications to download
            created_after: Optional ISO-8601 lower bound for created_at (inclusive)
            created_before: Optional ISO-8601 upper bound for created_at (inclusive)

        Returns:
            List of classification dictionaries containing volunteer annotations
        """
        from panoptes_client import Panoptes, Project, Classification

        logger.info("Downloading classifications using Panoptes client")
        logger.info("Note: API queries may not return all classifications. For complete data,")
        logger.info("use the data export feature from the Zooniverse project page.")

        # Connect to Panoptes
        if self.username and self.password:
            Panoptes.connect(username=self.username, password=self.password)
        else:
            Panoptes.connect()

        # Get project to find workflows if not specified
        if workflow_id is None:
            project = Project.find(self.BACKYARD_WORLDS_PROJECT_ID)
            workflows = project.links.workflows
            if workflows:
                workflow_id = workflows[0].id
                logger.info(f"Using workflow ID: {workflow_id}")

        classifications_data = []

        def _parse_iso(ts: Optional[str]):
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None

        created_after_dt = _parse_iso(created_after)
        created_before_dt = _parse_iso(created_before)

        def _in_date_window(created_at: Optional[str]) -> bool:
            if not created_at or (created_after_dt is None and created_before_dt is None):
                return True
            created_dt = _parse_iso(created_at)
            if created_dt is None:
                return True
            if created_after_dt and created_dt < created_after_dt:
                return False
            if created_before_dt and created_dt > created_before_dt:
                return False
            return True

        if subject_ids:
            logger.info(
                "Fetching project classifications and filtering for "
                f"{len(subject_ids)} specific subjects"
            )
            # Query project scope and filter locally for subject_ids
            try:
                query_params = {'scope': 'project', 'project_id': self.BACKYARD_WORLDS_PROJECT_ID}
                if workflow_id:
                    query_params['workflow_id'] = workflow_id

                classifications = Classification.where(**query_params)

                subject_ids_set = set(int(sid) for sid in subject_ids)
                for classification in tqdm(classifications, desc="Fetching classifications"):
                    if limit and len(classifications_data) >= limit:
                        break

                    try:
                        classification_subject_ids = classification.raw.get('links', {}).get('subjects', [])
                        if not classification_subject_ids:
                            continue

                        if not _in_date_window(classification.raw.get('created_at')):
                            continue

                        if any(int(sid) in subject_ids_set for sid in classification_subject_ids):
                            classifications_data.append(
                                {
                                    'classification_id': classification.id,
                                    'subject_ids': classification_subject_ids,
                                    'workflow_id': classification.raw.get('links', {}).get('workflow'),
                                    'user_id': classification.raw.get('links', {}).get('user'),
                                    'created_at': classification.raw.get('created_at'),
                                    'annotations': classification.raw.get('annotations', []),
                                    'metadata': classification.raw.get('metadata', {}),
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Failed to parse classification {classification.id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to fetch classifications: {e}")

        else:
            # Download all classifications for the project
            logger.info("Fetching all classifications for project")
            query_params = {'project_id': self.BACKYARD_WORLDS_PROJECT_ID}
            if workflow_id:
                query_params['workflow_id'] = workflow_id

            try:
                classifications = Classification.where(**query_params)

                for idx, classification in enumerate(
                    tqdm(classifications, desc="Downloading classifications", total=limit)
                ):
                    if limit and idx >= limit:
                        break

                    try:
                        # Extract subject IDs from links
                        subject_ids_in_classification = classification.raw.get('links', {}).get('subjects', [])
                        if not _in_date_window(classification.raw.get('created_at')):
                            continue

                        classifications_data.append(
                            {
                                'classification_id': classification.id,
                                'subject_ids': subject_ids_in_classification,
                                'workflow_id': classification.raw.get('links', {}).get('workflow'),
                                'user_id': classification.raw.get('links', {}).get('user'),
                                'created_at': classification.raw.get('created_at'),
                                'annotations': classification.raw.get('annotations', []),
                                'metadata': classification.raw.get('metadata', {}),
                            }
                        )

                    except Exception as e:
                        logger.warning(f"Failed to parse classification {classification.id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to fetch classifications: {e}")

        logger.info(f"Downloaded {len(classifications_data)} classifications")
        return classifications_data

    def request_data_export(self) -> Dict:
        """
        Request a data export from Zooniverse for the Backyard Worlds project.

        This generates a complete export of classifications which can be downloaded once ready.
        Requires authentication with project owner/collaborator credentials.

        Returns:
            Dictionary with export request information
        """
        from panoptes_client import Panoptes, Project

        logger.info("Requesting data export from Zooniverse")

        # Must be authenticated
        if not (self.username and self.password):
            raise ValueError("Data export requires authentication (username and password)")

        Panoptes.connect(username=self.username, password=self.password)

        # Get project
        project = Project.find(self.BACKYARD_WORLDS_PROJECT_ID)

        # Request export
        try:
            export = project.get_export(
                'classifications',
                generate=True,
                wait=False,
                wait_timeout=300,
            )
            logger.info("Export requested successfully!")
            logger.info("Check the project's Data Exports page to download when ready:")
            logger.info(f"https://www.zooniverse.org/lab/{project.id}/data-exports")

            return {
                'status': 'requested',
                'project_id': self.BACKYARD_WORLDS_PROJECT_ID,
                'export_url': f"https://www.zooniverse.org/lab/{project.id}/data-exports",
            }

        except Exception as e:
            logger.error(f"Failed to request export: {e}")
            logger.info(
                "You may need project owner/collaborator permissions to request data exports"
            )
            logger.info(
                f"Alternatively, download from: https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9/data-exports"
            )
            raise

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
