# %%

import os

import dask.array as da
import napari
from dask.distributed import Client, LocalCluster

# %%

if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    # slicer = (
    #     slice(None, None, 3),
    #     slice(None, None, 12),
    #     slice(None, None, 12),
    # )
    slicer = [
        slice(None),
        slice(None),
        slice(None),
    ]
    # Reading images
    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"))[*slicer].compute()
    arr_overlap = da.from_zarr(os.path.join(out_dir, "0_overlap.zarr"))[
        *slicer
    ].compute()
    arr_bgrm = da.from_zarr(os.path.join(out_dir, "1_bgrm.zarr"))[*slicer].compute()
    arr_dog = da.from_zarr(os.path.join(out_dir, "2_dog.zarr"))[*slicer].compute()
    arr_adaptv = da.from_zarr(os.path.join(out_dir, "3_adaptv.zarr"))[*slicer].compute()
    # arr_threshd = da.from_zarr(os.path.join(out_dir, "4_threshd.zarr"))[*slicer].compute()
    arr_sizes = da.from_zarr(os.path.join(out_dir, "5_sizes.zarr"))[*slicer].compute()
    arr_filt = da.from_zarr(os.path.join(out_dir, "6_filt.zarr"))[*slicer].compute()

    client.close()
    cluster.close()

    add_img_f = lambda arr, vmax: viewer.add_image(
        arr,
        # name=arr.__name__,
        contrast_limits=(0, vmax),
        blending="additive",
    )
    # Napari viewer adding images
    viewer = napari.Viewer()
    # img_raw = add_img_f(arr_raw, 10000)
    img_overlap = add_img_f(arr_overlap, 10000)
    img_bgrm = add_img_f(arr_bgrm, 500)
    img_dog = add_img_f(arr_dog, 100)
    img_adaptv = add_img_f(arr_adaptv, 100)
    # img_threshd = add_img_f(arr_threshd, 1)
    img_sizes = add_img_f(arr_sizes, 10000)
    img_filt = add_img_f(arr_filt, 1)

    napari.run()
