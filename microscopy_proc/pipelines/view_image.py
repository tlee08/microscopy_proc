# %%

import os

import dask.array as da
import napari
from dask.distributed import Client, LocalCluster

# %%

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    arr = da.from_zarr(os.path.join(out_dir, "0_raw.zarr"))
    arr = arr[::3, ::12, ::12]
    arr = arr.compute()

    viewer = napari.Viewer()
    img_layer = viewer.add_image(arr, multiscale=False)

    napari.run()

    client.close()
    cluster.close()
