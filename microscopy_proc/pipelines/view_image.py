# %%

import os

import dask.array as da
import napari
from dask.distributed import Client, LocalCluster

# %%


def add_img_f(viewer, arr, vmax):
    viewer.add_image(
        arr,
        # name=arr.__name__,
        contrast_limits=(0, vmax),
        blending="additive",
    )


def imgs1_f(out_dir):
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
    arr_overlap = da.from_zarr(os.path.join(out_dir, "0_overlap.zarr"))[
        *slicer
    ].compute()
    arr_bgrm = da.from_zarr(os.path.join(out_dir, "1_bgrm.zarr"))[*slicer].compute()
    arr_dog = da.from_zarr(os.path.join(out_dir, "2_dog.zarr"))[*slicer].compute()
    arr_adaptv = da.from_zarr(os.path.join(out_dir, "3_adaptv.zarr"))[*slicer].compute()
    # arr_threshd = da.from_zarr(os.path.join(out_dir, "4_threshd.zarr"))[*slicer].compute()
    arr_sizes = da.from_zarr(os.path.join(out_dir, "5_sizes.zarr"))[*slicer].compute()
    arr_filt = da.from_zarr(os.path.join(out_dir, "6_filt.zarr"))[*slicer].compute()
    arr_maxima = da.from_zarr(os.path.join(out_dir, "7_maxima.zarr"))[*slicer].compute()

    client.close()
    cluster.close()

    # Napari viewer adding images
    viewer = napari.Viewer()
    # add_img_f(viewer, arr_raw, 10000)
    add_img_f(viewer, arr_overlap, 10000)
    add_img_f(viewer, arr_bgrm, 2000)
    add_img_f(viewer, arr_dog, 100)
    add_img_f(viewer, arr_adaptv, 100)
    # add_img_f(viewer, arr_threshd, 5)
    add_img_f(viewer, arr_sizes, 10000)
    add_img_f(viewer, arr_filt, 5)
    add_img_f(viewer, arr_maxima, 5)

    napari.run()


def imgs2_f(out_dir):
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
    arr_filt_f = da.from_zarr(os.path.join(out_dir, "9_filt_f.zarr"))[*slicer].compute()
    arr_maxima_f = da.from_zarr(os.path.join(out_dir, "9_maxima_f.zarr"))[
        *slicer
    ].compute()

    client.close()
    cluster.close()

    # Napari viewer adding images
    viewer = napari.Viewer()
    add_img_f(viewer, arr_raw, 10000)
    add_img_f(viewer, arr_filt_f, 5)
    add_img_f(viewer, arr_maxima_f, 5)

    napari.run()
    
def imgs3_f(out_dir):
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    slicer = (
        slice(None, None, 3),
        slice(None, None, 12),
        slice(None, None, 12),
    )
    # Reading images
    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"))[*slicer].compute()
    # arr_filt_f = da.from_zarr(os.path.join(out_dir, "9_filt_f.zarr"))[*slicer].compute()
    # arr_maxima_f = da.from_zarr(os.path.join(out_dir, "9_maxima_f.zarr"))[
    #     *slicer
    # ].compute()

    client.close()
    cluster.close()

    # Napari viewer adding images
    viewer = napari.Viewer()
    add_img_f(viewer, arr_raw, 10000)
    # add_img_f(viewer, arr_filt_f, 5)
    # add_img_f(viewer, arr_maxima_f, 5)

    napari.run()


if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # imgs1_f(out_dir)
    imgs3_f(out_dir)
