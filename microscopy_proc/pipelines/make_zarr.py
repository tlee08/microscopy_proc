import os

from dask.distributed import LocalCluster

# from prefect import flow
from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.funcs.io_funcs import btiff_to_zarr, tiffs_to_zarr
from microscopy_proc.utils.dask_utils import cluster_proc_dec
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs


@cluster_proc_dec(lambda: LocalCluster())
# @flow
def tiff_to_zarr(in_fp, out_fp):
    if os.path.isdir(in_fp):
        tiffs_to_zarr(
            [os.path.join(in_fp, f) for f in os.listdir(in_fp)],
            out_fp,
            chunks=PROC_CHUNKS,
        )
    elif os.path.isfile(in_fp):
        btiff_to_zarr(
            in_fp,
            out_fp,
            chunks=PROC_CHUNKS,
        )
    else:
        raise ValueError("Input file path does not exist.")


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/example"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    tiff_to_zarr(in_fp, proj_fp_dict["raw"])
