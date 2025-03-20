import argparse
import glob
import logging
import os

import ocpmodels

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS = {
    "s2ef": {
        "200k": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
        "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
        "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
        "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz",
        "rattled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar",
        "md": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar",
    },
    "is2re": {
        "original": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
        "Oabs": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/0.tar",
    }
}

def get_data(datadir, task, split, del_intmd_files):
    os.makedirs(datadir, exist_ok=True)

    if task == "s2ef" and split is None:
        raise NotImplementedError("S2EF requires a split to be defined.")

    assert (
        split in DOWNLOAD_LINKS[task]
    ), f'"{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS["s2ef"].keys())}'
    download_link = DOWNLOAD_LINKS[task][split]

    os.system(f"wget {download_link} -P {datadir}")
    filename = os.path.join(datadir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {datadir}")
    dirname = os.path.join(
        datadir,
        os.path.basename(filename).split(".")[0],
    )
    compressed_dir = os.path.join(dirname, os.path.basename(dirname))
    output_path = os.path.join(datadir, "Oabs")
    uncompressed_dir = uncompress_data(compressed_dir)
    extract_last_frame(uncompressed_dir)
    write_txt_file(compressed_dir, uncompressed_dir)
    preprocess_data(uncompressed_dir, output_path)

    if del_intmd_files:
        cleanup(filename, dirname)


def uncompress_data(compressed_dir):
    import uncompress

    parser = uncompress.get_parser()
    args, _ = parser.parse_known_args()
    args.ipdir = compressed_dir
    args.opdir = os.path.dirname(compressed_dir) + "_uncompressed"
    uncompress.main(args)
    return args.opdir


def extract_last_frame(uncompressed_dir):
    for filename in os.listdir(uncompressed_dir):
        if filename.endswith('.extxyz'):
            file_path = os.path.join(uncompressed_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            frames = []
            i = 0
            while i < len(lines):
                try:
                    n_atoms = int(lines[i].strip())
                except ValueError:
                    print(f"Skipping file {filename}: unable to interpret the number of atoms on line {i+1}.")
                    break
                frame_size = n_atoms + 2
                if i + frame_size > len(lines):
                    print(f"Incomplete frame in {filename} starting at line {i+1}; skipping remainder.")
                    break
                frame = lines[i:i+frame_size]
                frames.append(frame)
                i += frame_size
            
            if frames:
                last_frame = frames[-1]
                with open(file_path, 'w') as f:
                    f.writelines(last_frame)
                # print(f"Processed {filename}: kept the last frame.")
            else:
                print(f"No valid frames found in {filename}.")


def write_txt_file(compressed_dir, uncompressed_dir):
    with open(os.path.join(compressed_dir, '../', 'system.txt'), 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split(',')
            file_base = parts[0]
            output_path = os.path.join(uncompressed_dir, f"{file_base}.txt")
            with open(output_path, 'w') as outfile:
                outfile.write(f'{parts[0]},frame0,{parts[1]}')
    logging.info(f"Writing to {uncompressed_dir}")
       
                
def preprocess_data(uncompressed_dir, output_path):
    import preprocess_last as preprocess

    parser = preprocess.get_parser()
    args, _ = parser.parse_known_args()
    args.data_path = uncompressed_dir
    args.out_path = output_path
    preprocess.main(args)


def cleanup(filename, dirname):
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if os.path.exists(dirname + "_uncompressed"):
        shutil.rmtree(dirname + "_uncompressed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to download")
    parser.add_argument(
        "--split", type=str, help="Corresponding data split to download"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate directories and files upon data retrieval/processing",
    )
    # Flags for S2EF train/val set preprocessing:
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'ocpmodels/data'",
    )
    args, _ = parser.parse_known_args()
    get_data(
        datadir=args.data_path,
        task=args.task,
        split=args.split,
        del_intmd_files=not args.keep,
    )
