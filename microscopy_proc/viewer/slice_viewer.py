# %%

import os

import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns

# %%


def make_maxima_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.scatterplot(x=df["x"], y=df["y"], marker=".", alpha=0.2, s=10, ax=ax)
    ax.invert_yaxis()


def make_img(arr, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, cmap="grey", **kwargs)
    ax.axis("off")


# %%


if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    slicer = (
        # slice(300, 400, None), #  slice(None, None, 3),
        100,
        slice(0, 5000, None),  #  slice(None, None, 12),
        slice(0, 4000, None),  #  slice(None, None, 12),
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
        # ("9_filt_f", 5),
        # ("9_maxima_f", 5),
        # ("points", 5),
        ("heatmaps", 1),
    )

    fp_ls = [os.path.join(out_dir, f"{i}.zarr") for i, j in imgs_ls]
    vmax_ls = [j for i, j in imgs_ls]

    for fp, vmax in zip(fp_ls, vmax_ls):
        arr = da.from_zarr(fp)[*slicer].compute()
        make_img(arr, vmin=0, vmax=vmax)


# %%
