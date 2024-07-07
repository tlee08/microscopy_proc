import os
import re

import numpy as np
from natsort import natsorted

# TODO: add request functionality to download Allen Mouse Atlas image:
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

#####################################################################
#                     Getting filepaths in order
#####################################################################


def get_fps(dir, pattern):
    """
    Looks in dir for filepaths with the pattern.
    The pattern follows the regex search syntax:
        - Write the pattern identical to the filenames, except for the Z0000 number.
        - Use `*` to indicate "any character any number of times" (i.e. for the Z number)
    An example pattern is:
    ```
    `r"Sample_11_zoom0.52_2.5x_dual_side_fusion_2x4 vertical-stitched_T001_Z(\d+?)_C01.tif"`
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
        print(fp)
        fp_new = re.sub(
            pattern,
            lambda x: x.group(0).zfill(4),
            fp,
        )
        print(fp_new)
        # os.rename(os.path.join(dir, fp), os.path.join(dir, new_fp))
        print()


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
    arr = np.load(fp, mmap_mode="r")
    # Making header contents
    header_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
DimSize = {arr.shape[2]} {arr.shape[1]} {arr.shape[0]}
HeaderSize = {get_npy_header_size(fp)}
ElementType = {dtype_mapper[str(arr.dtype)]}
ElementDataFile = {os.path.split(fp)[1]}
"""
    # Saving header file
    header_fp = f"{fp}.mhd"
    with open(header_fp, "w") as f:
        f.write(header_content)
    return


#####################################################################
#           Generating plots to compare images superimposed
#####################################################################


# def compare_imgs(ims, **kwargs):
#     """
#     Compares two images superimposes onto eachother.
#     Used to visually hand-check for registration process.

#     **kwargs are passed to `Axes.set` method
#     """
#     # Initialising fig
#     fig, ax = plt.subplots()
#     # List of cmaps
#     cmaps = [
#         # "inferno",
#         # "viridis",
#         # "plasma",
#         # "cividis",
#         "Reds",
#         "Greens",
#         "Blues",
#         "Greys_r",
#         "Purples",
#     ]
#     # Making images
#     for i, im in enumerate(ims):
#         ax.imshow(
#             im,
#             cmap=cmaps[i],
#             alpha=0.5,
#         )
#     ax.set_xlim(0, np.max([i.shape[1] for i in ims]))
#     ax.set_ylim(0, np.max([i.shape[0] for i in ims]))
#     ax.set(**kwargs)
#     return fig, ax


#####################################################################
#             Annotation Ontology File Handler Functions
#####################################################################


def silentremove(fp):
    try:
        os.remove(fp)
    except OSError:
        pass
