import json
import os
import re
import shutil

import numpy as np
from natsort import natsorted

from microscopy_proc.utils.logging_utils import init_logger

# TODO: add request functionality to download Allen Mouse Atlas image:
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

#####################################################################
#                     Getting filepaths in order
#####################################################################

logger = init_logger(__name__)


def get_fps(dir, pattern):
    """
    Looks in dir for filepaths with the pattern.
    The pattern follows the regex search syntax:
        - Write the pattern identical to the filenames, except for the Z0000 number.
        - Use `*` to indicate "any character any number of times" (i.e. for the Z number)
    An example pattern is:
    ```
    `r"Sample_11_zoom0.52_2.5x_dual_side_fusion_2x4 vertical-stitched_T001_Z(\\d+?)_C01.tif"`
    ```
    """
    fps_all = natsorted(os.listdir(dir))
    fps = [os.path.join(dir, i) for i in fps_all if re.search(pattern, i)]
    return fps


def rename_slices_fps(dir, pattern):
    """
    TODO: make pattern modular, instead of current (pref)(to_change)(suffix) setup
    Currently just converts slices filenames to <Z,4>.
    """

    for fp in os.listdir(dir):
        logger.debug(fp)
        fp_new = re.sub(
            pattern,
            lambda x: x.group(0).zfill(4),
            fp,
        )
        logger.debug(fp_new)
        # os.rename(os.path.join(dir, fp), os.path.join(dir, new_fp))


#####################################################################
#                     Make npy headers for ImageJ
#####################################################################


def get_npy_header_size(fp):
    with open(fp, "rb") as f:
        h_size = 0
        while True:
            char = f.read(1)
            h_size += 1
            if char == b"\n":
                break
    return h_size


def make_npy_header(fp):
    """
    Makes a npy mhd header file so ImageJ can read .npy spatial arrays.
    If `fp` is does not have the `.npy` file extension, then adds it.
    """
    # Adding ".npy" extension if missing
    if not re.search(r"\.npy$", fp):
        fp = f"{fp}.npy"
    # Making datatype name mapper
    dtype_mapper = {
        "int8": "MET_CHAR",
        "uint8": "MET_UCHAR",
        "int16": "MET_SHORT",
        "uint16": "MET_USHORT",
        "int32": "MET_INT",
        "uint32": "MET_UINT",
        "int64": "MET_LONG",
        "uint64": "MET_ULONG",
        "float32": "MET_FLOAT",
        "float64": "MET_DOUBLE",
    }

    # Loading array
    ar = np.load(fp, mmap_mode="r")
    # Making header contents
    header_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
DimSize = {ar.shape[2]} {ar.shape[1]} {ar.shape[0]}
HeaderSize = {get_npy_header_size(fp)}
ElementType = {dtype_mapper[str(ar.dtype)]}
ElementDataFile = {os.path.split(fp)[1]}
"""
    # Saving header file
    header_fp = f"{fp}.mhd"
    with open(header_fp, "w") as f:
        f.write(header_content)
    return


def read_json(fp: str) -> dict:
    with open(fp, "r") as f:
        return json.load(f)


def write_json(fp: str, data: dict) -> None:
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)


def silentremove(fp):
    if os.path.isfile(fp):
        try:
            os.remove(fp)
        except OSError:
            pass
    elif os.path.isdir(fp):
        try:
            shutil.rmtree(fp)
        except OSError:
            pass


def sanitise_smb_df(df):
    """
    Sanitizes the SMB share dataframe.
    Removes any column called "smb-share:server".
    """
    if "smb-share:server" in df.columns:
        df = df.drop(columns="smb-share:server")
    return df
