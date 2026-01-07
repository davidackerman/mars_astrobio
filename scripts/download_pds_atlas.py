#!/usr/bin/env python3
"""
Efficient parallel downloader for PDS Atlas image data.
Reads URLs from the curl script and downloads them in parallel with resume capability.

Can also optionally convert PDS Atlas URLs to NASA raw images URLs for EDR (raw) data.
"""

import re
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import time

# Configuration
MAX_WORKERS = 8  # Number of parallel downloads
CHUNK_SIZE = 8192  # Download chunk size in bytes
MAX_RETRIES = 3  # Number of retry attempts for failed downloads
TIMEOUT = 30  # Request timeout in seconds


def convert_to_raw_url(pds_atlas_url):
    """
    Convert PDS Atlas API URL to NASA raw images direct URL.

    WARNING: NASA's raw images server has inconsistent naming - sometimes uses
    SIF_ (RDR) filenames in /edr/ paths. This function returns the URL but
    doesn't guarantee the file exists.

    PDS Atlas gives RDR (processed) images via API:
        https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/
        mars2020_imgops/browse/sol/01613/ids/rdr/shrlc/SIF_1613_...png::14

    NASA raw images path (sometimes works):
        https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/01613/ids/edr/
        browse/shrlc/SIF_1613_...png  (might keep SIF_ or convert to SI1_)

    Args:
        pds_atlas_url: URL from PDS Atlas curl script

    Returns:
        tuple: (nasa_raw_url, filename) or (None, None) if conversion fails
                Note: URL may not exist due to inconsistent NASA naming
    """
    # Extract components from PDS Atlas URL
    # Example: .../sol/01613/ids/rdr/shrlc/SIF_1613_0810171983_281RZS_N0790102SRLC08062_0000LMJ01.png::14
    match = re.search(r'/sol/(\d+)/ids/rdr/(\w+)/(SIF?_\d+_[^:]+\.png)', pds_atlas_url)
    if not match:
        return None, None

    sol, camera_type, filename = match.groups()

    # NASA's raw images server is inconsistent - sometimes keeps SIF_ in /edr/ paths
    # We'll keep the same filename to match what NASA actually serves
    nasa_filename = filename

    # Build NASA raw images URL
    # Note: Using /edr/ path but keeping original filename due to NASA inconsistency
    nasa_url = (
        f"https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/"
        f"sol/{sol}/ids/edr/browse/{camera_type}/{nasa_filename}"
    )

    return nasa_url, nasa_filename


def parse_curl_script(script_path):
    """
    Parse the curl script file and extract URLs.

    Returns:
        List of tuples (url, filename)
    """
    urls = []
    url_pattern = re.compile(r'curl .+ (https://[^\s]+)')

    print(f"Parsing {script_path}...")
    with open(script_path, 'r') as f:
        for line in f:
            match = url_pattern.search(line)
            if match:
                url = match.group(1)
                # Extract filename from URL
                filename = url.split('/')[-1].split('::')[0]  # Remove ::14 suffix
                urls.append((url, filename))

    print(f"Found {len(urls)} URLs to download")
    return urls


def extract_sol_from_filename(filename):
    """
    Extract sol number from filename.
    Example: SIF_1613_0810171983_281RZS_N0790102SRLC08062_0000LMJ01.png -> sol_1613

    Returns:
        tuple: (sol_dir_name, sol_number) e.g., ("sol_1613", 1613)
    """
    # Look for pattern like _NNNN_ where NNNN is the sol number
    match = re.search(r'_(\d{4})_', filename)
    if match:
        sol_num = int(match.group(1))
        return f"sol_{sol_num:04d}", sol_num
    return "unknown_sol", None


def parse_sol_filter(sol_filter_str):
    """
    Parse sol filter string into a set of allowed sol numbers.

    Supported formats:
    - Single sol: "1212"
    - Comma-separated: "495,1212,1218"
    - Range: "1200-1220"
    - Multiple ranges: "495-510,1200-1220"
    - Named presets: "cheyava", "wildcat", "biosig" (all biosignature sites)

    Returns:
        set: Set of sol numbers to include, or None for all sols
    """
    if not sol_filter_str:
        return None

    # Named presets for known biosignature sites
    presets = {
        'cheyava': list(range(1200, 1221)),  # Cheyava Falls discovery (sol 1212-1218 + buffer)
        'wildcat': list(range(490, 511)),     # Wildcat Ridge organics (sol ~495-500 + buffer)
        'biosig': list(range(490, 511)) + list(range(1200, 1221)),  # All biosignature sites
    }

    if sol_filter_str.lower() in presets:
        return set(presets[sol_filter_str.lower()])

    allowed_sols = set()

    # Parse comma-separated values
    for part in sol_filter_str.split(','):
        part = part.strip()

        # Check for range (e.g., "1200-1220")
        if '-' in part:
            try:
                start, end = part.split('-')
                start_sol = int(start.strip())
                end_sol = int(end.strip())
                allowed_sols.update(range(start_sol, end_sol + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}', skipping")
                continue
        else:
            # Single sol number
            try:
                allowed_sols.add(int(part))
            except ValueError:
                print(f"Warning: Invalid sol number '{part}', skipping")
                continue

    return allowed_sols if allowed_sols else None


def download_file(url, output_path, max_retries=MAX_RETRIES):
    """
    Download a file with resume capability and retry logic.

    Args:
        url: URL to download
        output_path: Path where file should be saved
        max_retries: Maximum number of retry attempts

    Returns:
        tuple: (success: bool, message: str)
    """
    output_path = Path(output_path)

    # Check if file already exists and has content
    if output_path.exists() and output_path.stat().st_size > 0:
        # Verify file is complete by checking if we can get file size from server
        try:
            head_response = requests.head(url, timeout=TIMEOUT)
            expected_size = int(head_response.headers.get('content-length', 0))
            actual_size = output_path.stat().st_size

            if expected_size > 0 and actual_size == expected_size:
                return True, "Already downloaded"
        except:
            pass  # Continue with download if HEAD request fails

    # Create parent directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            # Download with streaming
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()

            # Write to temporary file first
            temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            # Move temp file to final location
            temp_path.rename(output_path)
            return True, "Downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return False, f"Failed after {max_retries} attempts: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    return False, "Unknown error"


def download_worker(task):
    """
    Worker function for parallel downloads.

    Args:
        task: tuple of (url, filename, output_dir)

    Returns:
        tuple: (filename, success, message)
    """
    url, filename, output_dir = task

    # Organize by sol number
    sol_dir, _ = extract_sol_from_filename(filename)
    output_path = Path(output_dir) / sol_dir / filename

    success, message = download_file(url, output_path)
    return filename, success, message


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python download_pds_atlas.py <curl_script_file> [output_dir] [max_workers] [OPTIONS]")
        print("\nArguments:")
        print("  curl_script_file  : Path to the PDS Atlas curl script (.bat file)")
        print("  output_dir        : Directory to save downloads (default: pds_atlas_downloads)")
        print("  max_workers       : Number of parallel downloads (default: 8)")
        print("\nOptions:")
        print("  --sols FILTER     : Filter by sol number(s)")
        print("  --use-raw         : Attempt to download from NASA raw images server instead of PDS Atlas")
        print("                      WARNING: NASA's server has inconsistent naming, may have errors")
        print("\nData Sources:")
        print("  PDS Atlas (default)  : Reliable RDR processed images via PDS API (RECOMMENDED)")
        print("  NASA Raw (--use-raw) : Direct from NASA server, inconsistent naming (EXPERIMENTAL)")
        print("\nSol Filter Examples:")
        print("  --sols cheyava           # Cheyava Falls biosignature site (sols 1200-1220)")
        print("  --sols wildcat           # Wildcat Ridge organics site (sols 490-510)")
        print("  --sols biosig            # All biosignature sites combined")
        print("  --sols 1212              # Single sol")
        print("  --sols 1200-1220         # Sol range")
        print("  --sols 495,1212,1218     # Multiple specific sols")
        print("  --sols 490-510,1200-1220 # Multiple ranges")
        print("\nExamples:")
        print("  python download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat")
        print("  python download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse")
        print("  python download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 16")
        print("  python download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_browse 8 --sols cheyava")
        print("  python download_pds_atlas.py pdsimg-atlas-curl_2026-01-06T01_11_53_944.bat data/raw/watson_raw 8 --sols cheyava --use-raw")
        sys.exit(1)

    script_path = sys.argv[1]
    output_dir = "pds_atlas_downloads"
    max_workers = MAX_WORKERS
    sol_filter_str = None
    use_raw = False

    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--sols':
            if i + 1 < len(sys.argv):
                sol_filter_str = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --sols requires a value")
                sys.exit(1)
        elif sys.argv[i] == '--use-raw':
            use_raw = True
            i += 1
        elif i == 2:
            output_dir = sys.argv[i]
            i += 1
        elif i == 3:
            try:
                max_workers = int(sys.argv[i])
            except ValueError:
                print(f"Error: max_workers must be an integer, got '{sys.argv[i]}'")
                sys.exit(1)
            i += 1
        else:
            print(f"Error: Unexpected argument '{sys.argv[i]}'")
            sys.exit(1)

    # Validate script file exists
    if not os.path.exists(script_path):
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    # Parse sol filter
    allowed_sols = parse_sol_filter(sol_filter_str)

    # Parse URLs from script
    urls = parse_curl_script(script_path)

    if not urls:
        print("No URLs found in script file!")
        sys.exit(1)

    # Convert to raw URLs if requested
    if use_raw:
        print("\nConverting PDS Atlas URLs to NASA raw images URLs (EDR)...")
        converted_urls = []
        failed_conversions = 0
        for url, filename in urls:
            raw_url, raw_filename = convert_to_raw_url(url)
            if raw_url:
                converted_urls.append((raw_url, raw_filename))
            else:
                # Keep original if conversion fails
                converted_urls.append((url, filename))
                failed_conversions += 1
        urls = converted_urls
        if failed_conversions > 0:
            print(f"Warning: Failed to convert {failed_conversions} URLs, using original URLs for those")
        print(f"Converted {len(urls) - failed_conversions} URLs to NASA raw images format")

    # Filter URLs by sol if requested
    if allowed_sols is not None:
        print(f"\nFiltering to {len(allowed_sols)} sol(s): {sorted(list(allowed_sols))[:10]}{'...' if len(allowed_sols) > 10 else ''}")
        original_count = len(urls)
        filtered_urls = []
        for url, filename in urls:
            _, sol_num = extract_sol_from_filename(filename)
            if sol_num is not None and sol_num in allowed_sols:
                filtered_urls.append((url, filename))
        urls = filtered_urls
        print(f"Filtered {original_count} URLs down to {len(urls)} URLs")

        if not urls:
            print("No URLs match the sol filter!")
            sys.exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nDownload configuration:")
    print(f"  URLs to download: {len(urls)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Max retries per file: {MAX_RETRIES}")
    print(f"  Data type: {'EDR (raw)' if use_raw else 'RDR (processed)'}")
    if allowed_sols:
        sol_range = f"sols {min(allowed_sols)}-{max(allowed_sols)}" if len(allowed_sols) > 3 else f"sol(s) {sorted(list(allowed_sols))}"
        print(f"  Sol filter: {sol_range}")
    print()

    # Create tasks for workers
    tasks = [(url, filename, output_dir) for url, filename in urls]

    # Download files in parallel with progress bar
    successful = 0
    failed = 0
    skipped = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(download_worker, task): task for task in tasks}

        # Process completed downloads with progress bar
        with tqdm(total=len(tasks), desc="Downloading", unit="file") as pbar:
            for future in as_completed(future_to_task):
                filename, success, message = future.result()

                if success:
                    if "Already downloaded" in message:
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1
                    failed_files.append((filename, message))

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'success': successful,
                    'skipped': skipped,
                    'failed': failed
                })

    # Print summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"Total files:       {len(urls)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Already existed:   {skipped}")
    print(f"Failed:            {failed}")
    print()

    # Print failed files if any
    if failed_files:
        print("Failed downloads:")
        for filename, message in failed_files[:20]:  # Show first 20
            print(f"  {filename}: {message}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")
        print()

        # Write failed files to a log
        log_path = Path(output_dir) / "failed_downloads.log"
        with open(log_path, 'w') as f:
            for filename, message in failed_files:
                f.write(f"{filename}\t{message}\n")
        print(f"Full list of failed downloads written to: {log_path}")

    print(f"\nFiles saved to: {output_dir}")
    print("Files organized by sol number in subdirectories (e.g., sol_1613/)")


if __name__ == "__main__":
    main()
