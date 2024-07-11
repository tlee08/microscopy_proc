# %%

import os

import dask.array as da
import napari
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
# %%

if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    slicer = (
        slice(None, None, 3),
        slice(None, None, 12),
        slice(None, None, 12),
    )

    imgs_ls = (
        "10_region",
        "10_maxima", 
    )
    fp_ls = [os.path.join(out_dir, f"{i}.parquet") for i in imgs_ls]
    df_ls = [dd.read_parquet(i) for i in fp_ls]

    
# %%
