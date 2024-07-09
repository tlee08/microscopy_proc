import os

from microscopy_proc.funcs.tiff_to_zarr import tiff_to_zarr

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    tiff_to_zarr(in_fp, os.path.join(out_dir, "raw.zarr"))
