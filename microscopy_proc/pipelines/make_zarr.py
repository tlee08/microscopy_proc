import os
import re

from dask.distributed import LocalCluster
from natsort import natsorted

# from prefect import flow
from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.funcs.io_funcs import btiff2zarr, tiffs2zarr
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    init_params,
    make_proj_dirs,
)


# @flow
def tiff2zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=6)):
        if os.path.isdir(in_fp):
            tiffs2zarr(
                natsorted(
                    [
                        os.path.join(in_fp, f)
                        for f in os.listdir(in_fp)
                        if re.search(r".tif$", f)
                    ]
                ),
                out_fp,
                chunks=chunks,
            )
        elif os.path.isfile(in_fp):
            btiff2zarr(
                in_fp,
                out_fp,
                chunks=chunks,
            )
        else:
            raise ValueError("Input file path does not exist.")


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    in_fp = "/home/linux1/Desktop/A-1-1/cropped_abcd.tif"
    # in_fp = "/home/linux1/Desktop/A-1-1/example"
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Making params json
    init_params(proj_fp_dict)

    tiff2zarr(in_fp, proj_fp_dict["raw"], chunks=PROC_CHUNKS)
