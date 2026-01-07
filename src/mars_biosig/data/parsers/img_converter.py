"""
PDS/VICAR IMG file converter.

Converts PDS IMG files (VICAR and PDS4 formats) to standard image formats (PNG, TIFF).
"""

import logging
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class IMGConverter:
    """
    Converter for PDS/VICAR IMG files to standard image formats.

    Handles raw IMG files from Mars rovers, particularly WATSON images.
    """

    # PDS4 XML namespaces
    NAMESPACES = {
        'pds': 'http://pds.nasa.gov/pds4/pds/v1',
        'img': 'http://pds.nasa.gov/pds4/img/v1',
    }

    def __init__(self):
        """Initialize IMG converter."""
        pass

    def read_pds4_label(self, xml_path: Path) -> dict:
        """
        Parse PDS4 XML label.

        Parameters
        ----------
        xml_path : Path
            Path to XML label file

        Returns
        -------
        dict
            Parsed label parameters including dimensions and offset
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        label = {}

        # Find File_Area_Observational
        file_area = root.find('.//pds:File_Area_Observational', self.NAMESPACES)
        if file_area is None:
            raise ValueError("Could not find File_Area_Observational in XML")

        # Try Array_2D_Image first, then Array_3D_Image
        array_elem = file_area.find('.//pds:Array_2D_Image', self.NAMESPACES)
        if array_elem is None:
            array_elem = file_area.find('.//pds:Array_3D_Image', self.NAMESPACES)
            if array_elem is not None:
                label['is_3d'] = True

        if array_elem is not None:
            # Get offset
            offset_elem = array_elem.find('pds:offset', self.NAMESPACES)
            if offset_elem is not None:
                label['offset'] = int(offset_elem.text)

            # Get axes info
            axes_elem = array_elem.find('pds:axes', self.NAMESPACES)
            if axes_elem is not None:
                label['axes'] = int(axes_elem.text)

            # Find Axis_Array elements
            axis_arrays = array_elem.findall('.//pds:Axis_Array', self.NAMESPACES)
            for axis in axis_arrays:
                axis_name = axis.find('pds:axis_name', self.NAMESPACES)
                elements = axis.find('pds:elements', self.NAMESPACES)
                if axis_name is not None and elements is not None:
                    if axis_name.text == 'Line':
                        label['lines'] = int(elements.text)
                    elif axis_name.text == 'Sample':
                        label['samples'] = int(elements.text)
                    elif axis_name.text == 'Band':
                        label['bands'] = int(elements.text)

            # Get data type
            element_array = array_elem.find('.//pds:Element_Array', self.NAMESPACES)
            if element_array is not None:
                data_type_elem = element_array.find('pds:data_type', self.NAMESPACES)
                if data_type_elem is not None:
                    label['data_type'] = data_type_elem.text

        logger.debug(f"Parsed PDS4 label: {label}")
        return label

    def read_vicar_label(self, img_path: Path) -> dict:
        """
        Parse VICAR label from IMG file.

        Parameters
        ----------
        img_path : Path
            Path to IMG file

        Returns
        -------
        dict
            Parsed label parameters
        """
        label = {}

        with open(img_path, 'rb') as f:
            # Read first chunk to find label
            header = f.read(10000).decode('latin-1', errors='ignore')

            # Extract key parameters using simple parsing
            for line in header.split('\n'):
                if '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove trailing comments
                        if '/' in value:
                            value = value.split('/')[0].strip()

                        # Clean up value
                        value = value.strip("'\" ")

                        # Convert numeric values
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except (ValueError, AttributeError):
                            pass

                        label[key] = value
                    except Exception:
                        continue

        logger.debug(f"Parsed VICAR label: {len(label)} parameters")
        return label

    def convert_img_to_array(
        self,
        img_path: Path,
        label: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Convert IMG file to numpy array.

        Parameters
        ----------
        img_path : Path
            Path to IMG file
        label : dict, optional
            Pre-parsed label. If None, will try PDS4 XML first, then VICAR.

        Returns
        -------
        tuple of (ndarray, dict)
            Image array and label metadata
        """
        img_path = Path(img_path)

        # Parse label if not provided
        if label is None:
            # Try PDS4 XML label first
            xml_path = img_path.with_suffix('.xml')
            if xml_path.exists():
                logger.info(f"Using PDS4 XML label: {xml_path.name}")
                label = self.read_pds4_label(xml_path)
            else:
                # Fall back to VICAR label
                logger.info("Using VICAR label from IMG file")
                label = self.read_vicar_label(img_path)

        # Extract image dimensions (PDS4 or VICAR format)
        lines = label.get('lines', label.get('LINES', label.get('NL', 0)))
        line_samples = label.get('samples', label.get('LINE_SAMPLES', label.get('NS', 0)))

        # Determine data type
        data_type = label.get('data_type', label.get('SAMPLE_TYPE', 'UnsignedByte'))
        sample_bits = label.get('SAMPLE_BITS', 8)

        # Map PDS4 data types to sample bits and signedness
        if 'Byte' in data_type and '2' not in data_type:
            sample_bits = 8
        elif 'MSB2' in data_type or 'SignedMSB2' in data_type or 'UnsignedMSB2' in data_type:
            sample_bits = 16
        elif '16' in str(data_type):
            sample_bits = 16

        # Determine label size/offset (in bytes)
        lblsize = label.get('offset', label.get('LBLSIZE', 0))
        if lblsize == 0:
            # Try to find end of label
            with open(img_path, 'rb') as f:
                content = f.read(50000)
                # Look for common label terminators
                for marker in [b'END\n', b'END\r\n', b'END ']:
                    if marker in content:
                        lblsize = content.index(marker) + len(marker)
                        break

        logger.info(f"Image dimensions: {lines}x{line_samples}, {sample_bits}-bit")
        logger.info(f"Label size: {lblsize} bytes")

        # Determine dtype (handle signed/unsigned and byte order)
        if sample_bits == 8:
            # 8-bit is always unsigned for images
            dtype = np.uint8
            bytes_per_sample = 1
        elif sample_bits == 16:
            # Check if signed or unsigned, and byte order (MSB = big-endian)
            if 'Signed' in data_type and 'Unsigned' not in data_type:
                # Signed 16-bit big-endian
                dtype = np.dtype('>i2')  # big-endian int16
            else:
                # Unsigned 16-bit big-endian
                dtype = np.dtype('>u2')  # big-endian uint16
            bytes_per_sample = 2
        elif sample_bits == 32:
            if 'REAL' in data_type or 'Real' in data_type:
                dtype = np.dtype('>f4')  # big-endian float32
            else:
                dtype = np.dtype('>u4')  # big-endian uint32
            bytes_per_sample = 4
        else:
            raise ValueError(f"Unsupported sample_bits: {sample_bits}")

        # Check if 3D image (multi-band)
        bands = label.get('bands', 1)
        is_3d = label.get('is_3d', False) or bands > 1

        # Read image data
        expected_size = lines * line_samples * bands * bytes_per_sample

        with open(img_path, 'rb') as f:
            # Skip label
            f.seek(lblsize)

            # Read image data
            data = f.read(expected_size)

        if len(data) < expected_size:
            logger.warning(
                f"File truncated: expected {expected_size} bytes, got {len(data)}"
            )

        # Convert to numpy array
        try:
            arr = np.frombuffer(data, dtype=dtype)
            if is_3d:
                # Reshape as (bands, lines, samples) then transpose to (lines, samples, bands)
                arr = arr.reshape((bands, lines, line_samples))
                arr = np.transpose(arr, (1, 2, 0))
            else:
                arr = arr.reshape((lines, line_samples))
        except ValueError as e:
            logger.error(f"Failed to reshape array: {e}")
            # Try to reshape with available data
            available_pixels = len(data) // bytes_per_sample
            arr = np.frombuffer(data[:available_pixels * bytes_per_sample], dtype=dtype)
            if is_3d:
                total_pixels_per_band = available_pixels // bands
                new_lines = total_pixels_per_band // line_samples
                arr = arr.reshape((bands, new_lines, line_samples))
                arr = np.transpose(arr, (1, 2, 0))
                logger.warning(f"Reshaped to {new_lines}x{line_samples}x{bands}")
            else:
                new_lines = available_pixels // line_samples
                arr = arr.reshape((new_lines, line_samples))
                logger.warning(f"Reshaped to {new_lines}x{line_samples}")

        return arr, label

    def convert_img_to_png(
        self,
        img_path: Path,
        output_path: Optional[Path] = None,
        normalize: bool = True,
        stretch: str = 'linear',
    ) -> Path:
        """
        Convert IMG file to PNG.

        Parameters
        ----------
        img_path : Path
            Path to IMG file
        output_path : Path, optional
            Output PNG path. If None, uses same name with .png extension
        normalize : bool
            Whether to normalize intensity values (default: True)
        stretch : str
            Histogram stretch method: 'linear', 'sqrt', 'log' (default: 'linear')

        Returns
        -------
        Path
            Path to output PNG file

        Examples
        --------
        >>> converter = IMGConverter()
        >>> png_path = converter.convert_img_to_png('image.IMG')
        """
        img_path = Path(img_path)

        if output_path is None:
            output_path = img_path.with_suffix('.png')
        else:
            output_path = Path(output_path)

        logger.info(f"Converting {img_path.name} to PNG")

        # Read IMG as array
        arr, label = self.convert_img_to_array(img_path)

        # Apply stretch if requested
        if normalize:
            arr = self._apply_stretch(arr, method=stretch)

        # Convert to 8-bit for PNG
        if arr.dtype != np.uint8:
            # Normalize to 0-255 range
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

        # Create PIL Image and save
        # Check if grayscale or RGB
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode='L')  # Grayscale
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr, mode='RGB')  # RGB color
        elif arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], mode='L')  # Single band as grayscale
        else:
            # For other multi-band images, just use first band
            img = Image.fromarray(arr[:, :, 0], mode='L')

        img.save(output_path, 'PNG')

        logger.info(f"Saved PNG: {output_path}")
        return output_path

    def _apply_stretch(self, arr: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Apply histogram stretch to enhance contrast.

        Parameters
        ----------
        arr : ndarray
            Input image array (2D or 3D)
        method : str
            Stretch method: 'linear', 'sqrt', 'log'

        Returns
        -------
        ndarray
            Stretched array
        """
        arr = arr.astype(np.float32)

        if method == 'linear':
            # Clip to 2nd-98th percentile
            # Apply per channel for RGB images
            if arr.ndim == 3:
                for i in range(arr.shape[2]):
                    p2, p98 = np.percentile(arr[:, :, i], [2, 98])
                    arr[:, :, i] = np.clip(arr[:, :, i], p2, p98)
            else:
                p2, p98 = np.percentile(arr, [2, 98])
                arr = np.clip(arr, p2, p98)

        elif method == 'sqrt':
            # Square root stretch (enhances shadows)
            arr = arr - arr.min()
            arr = np.sqrt(arr)

        elif method == 'log':
            # Logarithmic stretch (strong shadow enhancement)
            arr = arr - arr.min() + 1
            arr = np.log(arr)

        return arr

    def batch_convert(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.IMG",
        **kwargs,
    ) -> list:
        """
        Batch convert IMG files to PNG.

        Parameters
        ----------
        input_dir : Path
            Directory containing IMG files
        output_dir : Path, optional
            Output directory. If None, uses input_dir
        pattern : str
            Glob pattern for IMG files (default: "*.IMG")
        **kwargs
            Additional arguments passed to convert_img_to_png()

        Returns
        -------
        list of Path
            Paths to converted PNG files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(input_dir.glob(pattern))
        logger.info(f"Found {len(img_files)} IMG files in {input_dir}")

        converted = []
        for img_file in img_files:
            try:
                output_path = output_dir / img_file.with_suffix('.png').name
                png_path = self.convert_img_to_png(
                    img_file,
                    output_path=output_path,
                    **kwargs
                )
                converted.append(png_path)
            except Exception as e:
                logger.error(f"Failed to convert {img_file.name}: {e}")
                continue

        logger.info(f"Converted {len(converted)}/{len(img_files)} files")
        return converted
