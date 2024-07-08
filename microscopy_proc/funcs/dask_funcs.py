import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np

# def get_maxima_block(block):
#     res = block
#     res = tophat_filter(res, 10.0)
#     res = dog_filter(res, 2.0, 5.0)
#     res = gaussian_subtraction_filter(res, 101)
#     res = mean_thresholding(res, 0.0)
#     res = label_objects(res)
#     df_sizes = get_sizes(res)
#     res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)
#     res_maxima = get_local_maxima(block, 10)
#     res_maxima = mask(res_maxima, res)
#     res_maxima = label_objects(res_maxima)
#     # res_watershed = watershed_segm(block, res_maxima, res)

#     # Converting back to uint8
#     res_maxima = manual_thresholding(res_maxima, 0)
#     # Returning
#     return res_maxima


# def get_region_block(block):
#     res = block
#     res = tophat_filter(res, 10.0)
#     res = dog_filter(res, 2.0, 5.0)
#     res = gaussian_subtraction_filter(res, 101)
#     res = mean_thresholding(res, 0.0)
#     res = label_objects(res)
#     df_sizes = get_sizes(res)
#     res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)

#     # Converting back to uint8
#     res = manual_thresholding(res, 0)
#     # Returning
#     return res


# def get_maxima_block_from_region(block_raw, block_region):
#     res = get_local_maxima(block_raw, 10)
#     res = mask(res, block_region)
#     res = label_objects(res)
#     # res_watershed = watershed_segm(block, res_maxima, res)

#     # Converting back to uint8
#     res = manual_thresholding(res, 0)
#     # Returning
#     return res


def block_to_coords(func, arr: da.Array) -> dd.DataFrame:
    """
    Applies the `func` to `arr`.
    Expects `func` to convert `arr` to coords df (of sorts).

    Importantly, this offsets the coords in each block.
    """
    inds = np.meshgrid(*[np.cumsum([0, *i[:-1]]) for i in arr.chunks], indexing="ij")

    @dask.delayed
    def func_offsetted(block, z_offset, y_offset, x_offset):
        df = func(block)
        df["z"] = df["z"] + z_offset if "z" in df.columns else z_offset
        df["y"] = df["y"] + y_offset if "y" in df.columns else y_offset
        df["x"] = df["x"] + x_offset if "x" in df.columns else x_offset
        return df

    return dd.from_delayed(
        [
            func_offsetted(block, i, j, k)
            for block, i, j, k in zip(
                arr.to_delayed().ravel(),
                inds[0].ravel(),
                inds[1].ravel(),
                inds[2].ravel(),
            )
        ]
    )


def coords_to_block(df: dd.DataFrame, block_info: dict) -> dd.DataFrame:
    """
    Converts the coords to a block.
    """
    # Getting block info
    z, y, x = block_info[0]["array-location"]
    # Copying df
    df = df.copy()
    # Offsetting
    df["z"] = df["z"] - z[0]
    df["y"] = df["y"] - y[0]
    df["x"] = df["x"] - x[0]
    # Returning df
    return df


def disk_cache(arr: da.Array, fp):
    arr.to_zarr(fp, overwrite=True)
    return da.from_zarr(fp)
