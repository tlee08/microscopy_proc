# %%

import os

import dask.array as da
import napari
from dask.distributed import Client, LocalCluster

# %%


def add_img(viewer, arr, vmax):
    viewer.add_image(
        arr,
        # name=arr.__name__,
        contrast_limits=(0, vmax),
        blending="additive",
    )


def view_imgs(fp_ls, vmax_ls, slicer):
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    arr_ls = [da.from_zarr(i)[*slicer].compute() for i in fp_ls]

    client.close()
    cluster.close()

    # Napari viewer adding images
    viewer = napari.Viewer()
    for i, j in zip(arr_ls, vmax_ls):
        add_img(viewer, i, j)

    napari.run()


if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    slicer = (
        slice(400, 450, None),  #  slice(None, None, 3),
        slice(1000, None, None),  #  slice(None, None, 12),
        slice(1000, 5000, None),  #  slice(None, None, 12),
    )

    imgs_ls = (
        ("raw", 10000),
        # ("0_overlap", 10000),
        # ("1_bgrm", 2000),
        # ("2_dog", 100),
        # ("3_adaptv", 100),
        # ("4_threshd", 5),
        # ("5_sizes", 10000),
        # ("6_filt", 5),
        # ("7_maxima", 5),
        ("9_filt_f", 5),
        ("9_maxima_f", 1),
        # ("points", 5),
        ("heatmaps", 5),
    )
    fp_ls = [os.path.join(out_dir, f"{i}.zarr") for i, j in imgs_ls]
    vmax_ls = [j for i, j in imgs_ls]

    view_imgs(fp_ls, vmax_ls, slicer)
