"""
NASA Planetary Data System (PDS) API client.

This module provides a client for accessing Mars 2020 mission data
from NASA's PDS Geosciences Node and Mars Photos API.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from mars_biosig.utils.config import get_config
    _has_config = True
except ImportError:
    _has_config = False

logger = logging.getLogger(__name__)


class PDSClient:
    """
    Client for NASA Planetary Data System API.

    Supports both the Mars Photos API and direct PDS archive access.
    Includes automatic retries, rate limiting, and caching.
    """

    # API endpoints
    MARS_PHOTOS_API = "https://api.nasa.gov/mars-photos/api/v1"
    PDS_ARCHIVE_BASE = "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission"
    PDS_IMAGING_BASE = "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_imgops"
    WATSON_DATA_PATH = "data_watson_imgops/sol"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: int = 30,
    ):
        """
        Initialize PDS client.

        Parameters
        ----------
        api_key : str, optional
            NASA API key. If not provided, reads from NASA_API_KEY env variable.
            Get your key at https://api.nasa.gov/
        cache_dir : Path, optional
            Directory for caching responses. Defaults to data/cache/
        max_retries : int
            Maximum number of retry attempts for failed requests
        backoff_factor : float
            Exponential backoff factor for retries (seconds)
        timeout : int
            Request timeout in seconds
        """
        # Try to load from config first, then environment, then default
        if api_key is None and _has_config:
            try:
                config = get_config()
                api_key = config.nasa_api_key
            except ValueError:
                # Config validation failed, fall back to env
                pass

        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        # Set up session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        logger.info(f"Initialized PDS client with API key: {self.api_key[:8]}...")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        use_api_key: bool = True,
    ) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.

        Parameters
        ----------
        url : str
            Request URL
        params : dict, optional
            Query parameters
        use_api_key : bool
            Whether to include API key in request

        Returns
        -------
        Response
            HTTP response object

        Raises
        ------
        requests.HTTPError
            If request fails after retries
        """
        self._rate_limit()

        if params is None:
            params = {}

        if use_api_key and self.api_key:
            params["api_key"] = self.api_key

        logger.debug(f"GET {url} with params {params}")

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def query_rover_photos(
        self,
        rover: str = "perseverance",
        sol: Optional[int] = None,
        earth_date: Optional[str] = None,
        camera: Optional[str] = None,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        Query Mars Photos API for rover images.

        Parameters
        ----------
        rover : str
            Rover name (default: "perseverance")
        sol : int, optional
            Martian sol (day). If provided, earth_date is ignored.
        earth_date : str, optional
            Earth date in YYYY-MM-DD format
        camera : str, optional
            Camera name (e.g., "WATSON", "ZCAM", "SHERLOC_WATSON")
        page : int
            Page number for pagination (default: 1)

        Returns
        -------
        dict
            JSON response with photos data

        Examples
        --------
        >>> client = PDSClient()
        >>> photos = client.query_rover_photos(sol=10, camera="WATSON")
        >>> print(f"Found {len(photos['photos'])} images")
        """
        if sol is None and earth_date is None:
            raise ValueError("Must provide either sol or earth_date")

        url = f"{self.MARS_PHOTOS_API}/rovers/{rover}/photos"

        params = {"page": page}
        if sol is not None:
            params["sol"] = sol
        elif earth_date is not None:
            params["earth_date"] = earth_date

        if camera:
            params["camera"] = camera

        response = self._make_request(url, params=params)
        return response.json()

    def get_rover_manifest(self, rover: str = "perseverance") -> Dict[str, Any]:
        """
        Get rover mission manifest (available sols, cameras, etc.).

        Parameters
        ----------
        rover : str
            Rover name (default: "perseverance")

        Returns
        -------
        dict
            Mission manifest data
        """
        url = f"{self.MARS_PHOTOS_API}/manifests/{rover}"
        response = self._make_request(url)
        return response.json()

    def download_file(
        self,
        url: str,
        output_path: Path,
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Download a file from URL to local path.

        Parameters
        ----------
        url : str
            File URL
        output_path : Path
            Local output path
        overwrite : bool
            Whether to overwrite existing file (default: False)
        show_progress : bool
            Whether to show download progress (default: True)

        Returns
        -------
        Path
            Path to downloaded file
        """
        output_path = Path(output_path)

        # Check if file already exists
        if output_path.exists() and not overwrite:
            logger.info(f"File already exists: {output_path}")
            return output_path

        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        self._rate_limit()
        logger.info(f"Downloading {url} to {output_path}")

        try:
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            # Get file size for progress
            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                if show_progress and total_size > 0:
                    downloaded = 0
                    chunk_size = 8192

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if show_progress:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end="", flush=True)

                    if show_progress:
                        print()  # New line after progress
                else:
                    f.write(response.content)

            logger.info(f"Downloaded {output_path.name} ({total_size} bytes)")
            return output_path

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            raise

    def get_pds_product_url(
        self,
        product_id: str,
        instrument: str = "watson",
    ) -> str:
        """
        Construct PDS archive URL for a product.

        Parameters
        ----------
        product_id : str
            PDS product ID (e.g., "NLF_0001_0000000001_001")
        instrument : str
            Instrument name

        Returns
        -------
        str
            Full URL to product in PDS archive
        """
        # This is a simplified version - actual PDS URLs are more complex
        # and depend on product type, processing level, etc.
        instrument_path = f"data_{instrument}"
        return urljoin(self.PDS_ARCHIVE_BASE, f"{instrument_path}/{product_id}")

    def list_watson_sols(self) -> List[int]:
        """
        List all available WATSON sols in the PDS archive.

        Returns
        -------
        list of int
            List of sol numbers with WATSON data available

        Examples
        --------
        >>> client = PDSClient()
        >>> sols = client.list_watson_sols()
        >>> print(f"Found {len(sols)} sols with WATSON data")
        """
        from bs4 import BeautifulSoup

        url = f"{self.PDS_IMAGING_BASE}/{self.WATSON_DATA_PATH}/"

        try:
            response = self._make_request(url, use_api_key=False)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract sol directories (format: 00530/)
            sols = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href and href.endswith('/') and href[:-1].isdigit():
                    sol_num = int(href[:-1])
                    sols.append(sol_num)

            logger.info(f"Found {len(sols)} sols with WATSON data")
            return sorted(sols)

        except Exception as e:
            logger.error(f"Failed to list WATSON sols: {e}")
            raise

    def list_watson_images(
        self,
        sol: int,
        data_type: str = "edr",
    ) -> List[Dict[str, str]]:
        """
        List WATSON IMG files for a specific sol.

        Parameters
        ----------
        sol : int
            Martian sol number
        data_type : str
            Data product type: 'edr' (raw), 'rdr' (reduced), or 'fdr' (formatted)
            Default: 'edr' (raw data)

        Returns
        -------
        list of dict
            List of dictionaries with 'filename', 'url', 'size', and 'label_url'

        Examples
        --------
        >>> client = PDSClient()
        >>> images = client.list_watson_images(sol=530)
        >>> print(f"Found {len(images)} images for sol 530")
        """
        from bs4 import BeautifulSoup

        # Construct URL: sol/00530/ids/edr/shrlc/
        sol_str = f"{sol:05d}"
        url = f"{self.PDS_IMAGING_BASE}/{self.WATSON_DATA_PATH}/{sol_str}/ids/{data_type}/shrlc/"

        try:
            response = self._make_request(url, use_api_key=False)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract IMG files and their metadata
            images = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.endswith('.IMG'):
                    # Get file size from parent row
                    parent = link.parent.parent
                    size_text = parent.find_all('td')[1].text.strip() if len(parent.find_all('td')) > 1 else 'unknown'

                    img_url = urljoin(url, href)
                    label_url = urljoin(url, href.replace('.IMG', '.xml'))

                    images.append({
                        'filename': href,
                        'url': img_url,
                        'size': size_text,
                        'label_url': label_url,
                        'sol': sol,
                        'data_type': data_type,
                    })

            logger.info(f"Found {len(images)} IMG files for sol {sol} ({data_type})")
            return images

        except Exception as e:
            logger.warning(f"Failed to list images for sol {sol}: {e}")
            return []

    def download_watson_image(
        self,
        image_info: Dict[str, str],
        output_dir: Path,
        download_label: bool = True,
        overwrite: bool = False,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Download a WATSON IMG file and its XML label.

        Parameters
        ----------
        image_info : dict
            Image metadata from list_watson_images()
        output_dir : Path
            Directory to save files
        download_label : bool
            Whether to also download the XML label (default: True)
        overwrite : bool
            Whether to overwrite existing files

        Returns
        -------
        tuple of (Path, Path)
            Paths to downloaded (image_file, label_file)
            Either can be None if download failed or was skipped

        Examples
        --------
        >>> client = PDSClient()
        >>> images = client.list_watson_images(sol=530)
        >>> img_path, label_path = client.download_watson_image(
        ...     images[0], Path("data/raw/watson")
        ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download IMG file
        img_filename = image_info['filename']
        img_path = output_dir / img_filename

        try:
            img_path = self.download_file(
                image_info['url'],
                img_path,
                overwrite=overwrite,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"Failed to download {img_filename}: {e}")
            return None, None

        # Download label file
        label_path = None
        if download_label:
            label_filename = img_filename.replace('.IMG', '.xml')
            label_path = output_dir / label_filename

            try:
                label_path = self.download_file(
                    image_info['label_url'],
                    label_path,
                    overwrite=overwrite,
                    show_progress=False,
                )
            except Exception as e:
                logger.warning(f"Failed to download label {label_filename}: {e}")
                label_path = None

        return img_path, label_path

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
