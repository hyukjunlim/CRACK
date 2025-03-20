#!/usr/bin/env python
"""
This script downloads the O per‐adsorbate trajectories dataset,
extracts and decompresses the trajectory files, processes each file to
extract the final (relaxed) structure, and saves those structures into
a designated output directory.
"""

import os
import argparse
import logging
import tarfile
import glob
import lzma
from ase.io import read, write

# URL for the oxygen (O) per-adsorbate trajectories tarball
DATA_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/0.tar"

def download_file(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Downloading from {url}...")
    # Using wget for simplicity (could use requests if desired)
    os.system(f"wget {url} -P {out_dir}")
    local_file = os.path.join(out_dir, os.path.basename(url))
    logging.info(f"Downloaded file to: {local_file}")
    return local_file

def extract_tar(tar_path, extract_dir):
    logging.info(f"Extracting {tar_path} into {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    logging.info("Extraction complete.")
    return extract_dir

def decompress_xz_file(xz_path, out_path):
    """Decompress a single .xz file using the lzma module."""
    with lzma.open(xz_path, 'rb') as fin, open(out_path, 'wb') as fout:
        fout.write(fin.read())

def process_trajectory_files(input_dir, output_dir):
    """
    Process all .extxyz.xz files in input_dir by:
      1. Decompressing them to obtain .extxyz files.
      2. Reading the trajectory using ASE.
      3. Extracting the last frame (relaxed structure).
      4. Saving the final structure in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Recursively find all .extxyz.xz files
    traj_files = glob.glob(os.path.join(input_dir, "**", "*.extxyz.xz"), recursive=True)
    logging.info(f"Found {len(traj_files)} trajectory files to process.")
    
    for traj_file in traj_files:
        # Decompress file: remove .xz extension to form the decompressed filename.
        decompressed_file = traj_file[:-3]
        logging.info(f"Decompressing {traj_file}...")
        decompress_xz_file(traj_file, decompressed_file)
        
        # Read all frames in the trajectory; select the last (relaxed) frame.
        try:
            trajectory = read(decompressed_file, index=":")
            final_frame = trajectory[-1]
        except Exception as e:
            logging.error(f"Error reading {decompressed_file}: {e}")
            continue

        # Save the final frame to the output directory
        out_filename = os.path.basename(decompressed_file)
        out_path = os.path.join(output_dir, out_filename)
        write(out_path, final_frame)
        logging.info(f"Saved relaxed structure to {out_path}")
        
        # Optionally remove the decompressed file to save space
        os.remove(decompressed_file)
        
    logging.info("All trajectory files have been processed.")

def main():
    parser = argparse.ArgumentParser(
        description="Download, process, and build the O per-adsorbate trajectories dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where the tarball will be downloaded and processed."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_relaxed_structures",
        help="Directory where the relaxed (final) structures will be saved."
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the downloaded tar file after extraction."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Step 1: Download the dataset tarball.
    tar_file = download_file(DATA_URL, args.data_dir)

    # Step 2: Extract the tarball.
    extract_dir = os.path.join(args.data_dir, "per_adsorbate_O")
    extract_tar(tar_file, extract_dir)

    # Step 3: Process trajectory files.
    process_trajectory_files(extract_dir, args.output_dir)

    # Optional cleanup: remove tar file if --keep-tar is not set.
    if not args.keep_tar:
        logging.info("Removing downloaded tar file...")
        os.remove(tar_file)
    logging.info("Dataset processing completed successfully.")

if __name__ == "__main__":
    main()
