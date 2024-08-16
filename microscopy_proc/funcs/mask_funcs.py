import numpy as np
import pandas as pd

# logging.basicConfig(level=logging.DEBUG)
# from prefect import flow


def make_outline(arr: np.ndarray) -> pd.DataFrame:
    # Shifting along last axis
    l_shift = np.concatenate([arr[..., 1:], np.zeros((*arr.shape[:-1], 1))], axis=-1)
    r_shift = np.concatenate([np.zeros((*arr.shape[:-1], 1)), arr[..., :-1]], axis=-1)
    # Finding outline (ins and outs)
    coords_df = pd.concat(
        [
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (r_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(is_in=1),
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (l_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(is_in=0),
        ]
    )
    # Ordering by z, y, x, so fill outline works
    coords_df = coords_df.sort_values(by=["z", "y", "x"]).reset_index(drop=True)
    # Returning
    return coords_df


def fill_outline(arr: np.ndarray, coords_df: pd.DataFrame) -> np.ndarray:
    # Initialize mask
    res = np.zeros(arr.shape, dtype=np.uint8)
    # Checking that type is 0 or 1
    assert coords_df["is_in"].isin([0, 1]).all()
    # Ordering by z, y, x, so fill outline works
    coords_df = coords_df.sort_values(by=["z", "y", "x"]).reset_index(drop=True)
    # For each outline coord
    for i, x in coords_df.iterrows():
        # If type is 1, fill in (from current voxel)
        if x["is_in"] == 1:
            res[x["z"], x["y"], x["x"] :] = 1
        # If type is 0, stop filling in (after current voxel)
        elif x["is_in"] == 0:
            res[x["z"], x["y"], x["x"] + 1 :] = 0
    # Returning
    return res


def mask_to_region_counts(arr_mask: np.ndarray, arr_annot: np.ndarray) -> pd.DataFrame:
    # Convert arr_mask to binary
    arr_mask = (arr_mask > 0).astype(np.uint8)
    # Multiply mask by annotation to convert mask to region IDs
    arr_mask = arr_mask * arr_annot
    # Getting annotated region IDs
    id_labels, id_counts = np.unique(arr_mask, return_counts=True)
    # Returning region IDs and counts as dataframe
    return pd.DataFrame(
        {
            "id": id_labels,
            "volume": id_counts,
        }
    ).drop(index=0)
