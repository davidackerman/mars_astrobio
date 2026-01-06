"""
PDS4 (Planetary Data System version 4) XML label parser.

PDS4 labels contain metadata about Mars rover data products.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import xmltodict

logger = logging.getLogger(__name__)


class PDS4Parser:
    """
    Parser for PDS4 XML labels.

    PDS4 labels are XML files that accompany data products and contain
    detailed metadata about observations, instruments, and processing.
    """

    def __init__(self):
        """Initialize PDS4 parser."""
        pass

    def parse_label(self, label_path: Path) -> Dict[str, Any]:
        """
        Parse PDS4 XML label file.

        Parameters
        ----------
        label_path : Path
            Path to .xml label file

        Returns
        -------
        dict
            Parsed metadata dictionary

        Raises
        ------
        FileNotFoundError
            If label file doesn't exist
        ValueError
            If XML is invalid
        """
        label_path = Path(label_path)

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        logger.debug(f"Parsing PDS4 label: {label_path}")

        try:
            with open(label_path, "r") as f:
                xml_dict = xmltodict.parse(f.read())

            # Extract key metadata
            metadata = self._extract_metadata(xml_dict)
            return metadata

        except Exception as e:
            logger.error(f"Failed to parse {label_path}: {e}")
            raise ValueError(f"Invalid PDS4 XML: {e}")

    def _extract_metadata(self, xml_dict: Dict) -> Dict[str, Any]:
        """
        Extract relevant metadata from parsed XML.

        Parameters
        ----------
        xml_dict : dict
            Parsed XML dictionary from xmltodict

        Returns
        -------
        dict
            Cleaned metadata dictionary
        """
        # Navigate PDS4 structure (this is simplified - actual structure varies)
        product = xml_dict.get("Product_Observational", {})

        # Identification area
        identification = product.get("Identification_Area", {})

        # Observation area
        observation = product.get("Observation_Area", {})
        time_coords = observation.get("Time_Coordinates", {})
        investigation = observation.get("Investigation_Area", {})
        observing_system = observation.get("Observing_System", {})

        # File area
        file_area = product.get("File_Area_Observational", {})
        file_obj = file_area.get("File", {}) if file_area else {}

        metadata = {
            # Product identification
            "product_id": self._safe_get(identification, "logical_identifier"),
            "version_id": self._safe_get(identification, "version_id"),
            "title": self._safe_get(identification, "title"),
            "product_class": self._safe_get(identification, "product_class"),

            # Time information
            "start_time": self._parse_time(
                self._safe_get(time_coords, "start_date_time")
            ),
            "stop_time": self._parse_time(
                self._safe_get(time_coords, "stop_date_time")
            ),
            "local_mean_solar_time": self._safe_get(
                time_coords, "local_mean_solar_time"
            ),

            # Mission information
            "mission_name": self._safe_get(investigation, "name"),
            "mission_phase": self._safe_get(
                observation, "Mission_Area", "mission_phase_name"
            ),
            "sol": self._extract_sol(observation),

            # Instrument information
            "instrument_name": self._extract_instrument_name(observing_system),
            "instrument_id": self._extract_instrument_id(observing_system),

            # Target (e.g., Mars, specific rock)
            "target_name": self._safe_get(
                observation, "Target_Identification", "name"
            ),
            "target_type": self._safe_get(
                observation, "Target_Identification", "type"
            ),

            # File information
            "file_name": self._safe_get(file_obj, "file_name"),
            "file_size": self._safe_get(file_obj, "file_size"),

            # Processing information
            "processing_level": self._safe_get(
                identification, "Modification_History", "Modification_Detail", "description"
            ),
        }

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return metadata

    def _safe_get(self, d: Dict, *keys: str) -> Optional[Any]:
        """
        Safely navigate nested dictionary.

        Parameters
        ----------
        d : dict
            Dictionary to navigate
        *keys : str
            Keys to navigate

        Returns
        -------
        Any or None
            Value if found, None otherwise
        """
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key)
            else:
                return None
        return d

    def _parse_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """
        Parse PDS4 time string to datetime.

        Parameters
        ----------
        time_str : str or None
            ISO 8601 time string

        Returns
        -------
        datetime or None
            Parsed datetime object
        """
        if not time_str:
            return None

        try:
            # PDS4 uses ISO 8601 format
            # Example: "2021-02-18T20:12:33.184Z"
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            return datetime.fromisoformat(time_str)
        except Exception as e:
            logger.warning(f"Failed to parse time '{time_str}': {e}")
            return None

    def _extract_sol(self, observation: Dict) -> Optional[int]:
        """
        Extract sol number from observation area.

        Parameters
        ----------
        observation : dict
            Observation area dictionary

        Returns
        -------
        int or None
            Sol number if found
        """
        # Sol can be in different places in PDS4 labels
        mission_area = observation.get("Mission_Area", {})

        # Try different possible locations
        sol_keys = [
            "sol",
            "sol_number",
            "start_sol",
            "mars_sol_number",
        ]

        for key in sol_keys:
            sol = mission_area.get(key)
            if sol is not None:
                try:
                    return int(sol)
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_instrument_name(self, observing_system: Dict) -> Optional[str]:
        """Extract instrument name from observing system."""
        if not observing_system:
            return None

        # Instrument info can be nested
        components = observing_system.get("Observing_System_Component", [])

        # Handle both single component and list
        if not isinstance(components, list):
            components = [components]

        for component in components:
            comp_type = component.get("type", "").lower()
            if "instrument" in comp_type:
                return component.get("name")

        return None

    def _extract_instrument_id(self, observing_system: Dict) -> Optional[str]:
        """Extract instrument ID from observing system."""
        if not observing_system:
            return None

        components = observing_system.get("Observing_System_Component", [])

        if not isinstance(components, list):
            components = [components]

        for component in components:
            comp_type = component.get("type", "").lower()
            if "instrument" in comp_type:
                return component.get("Internal_Reference", {}).get("lid_reference")

        return None

    def parse_and_save(self, label_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Parse PDS4 label and save metadata as JSON.

        Parameters
        ----------
        label_path : Path
            Path to XML label file
        output_path : Path, optional
            Output JSON path. If None, uses same name with .json extension.

        Returns
        -------
        Path
            Path to saved JSON file
        """
        import json

        metadata = self.parse_label(label_path)

        if output_path is None:
            output_path = label_path.with_suffix(".metadata.json")

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved metadata to {output_path}")
        return output_path
