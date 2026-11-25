"""Download utility for SNAP Facebook dataset."""
import os
import tarfile
import urllib.request
from pathlib import Path

SNAP_FACEBOOK_URL = "http://snap.stanford.edu/data/facebook.tar.gz"

def download_and_extract_snap(
    raw_dir: str = "data/raw",
    url: str = SNAP_FACEBOOK_URL,
    force: bool = False
) -> Path:
    """Download facebook.tar.gz from Stanford SNAP and extract to raw_dir/facebook.
    
    Args:
        raw_dir: Directory to save archive and extract contents.
        url: URL of SNAP facebook archive.
        force: If True, re-download even if already extracted.
        
    Returns:
        Path to extracted facebook folder.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = raw_path / "facebook.tar.gz"
    extract_path = raw_path / "facebook"
    
    # Check if already extracted
    sample_file = extract_path / "0.edges"
    if extract_path.exists() and sample_file.exists() and not force:
        print(f"[download_snap] Dataset already exists at {extract_path}")
        return extract_path
        
    print(f"[download_snap] Downloading {url} to {tar_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response, open(tar_path, "wb") as out_file:
        chunk_size = 64 * 1024
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
            
    print(f"[download_snap] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_path)
        
    print(f"[download_snap] Successfully extracted to {extract_path}")
    return extract_path

if __name__ == "__main__":
    download_and_extract_snap()
