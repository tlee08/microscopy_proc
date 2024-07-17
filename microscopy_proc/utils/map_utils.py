import numpy as np
import pandas as pd


def get_coord_voxel(arr, z_val, y_val, x_val):
    """
    Given a 3d array (i.d. 3d image), and `(x, y, z)` coordinates that may be floats,
    returns the value of the nearest pixel in the 3d array.
    If the given coordinates are out of the 3d array range, then returns -1.
    """
    z = np.round(z_val, 0).astype(np.int32)
    y = np.round(y_val, 0).astype(np.int32)
    x = np.round(x_val, 0).astype(np.int32)
    if np.all((np.array([z, y, x]) >= 0) & (np.array([z, y, x]) < arr.shape)):
        return arr[z, y, x]
    else:
        return -1


def nested_tree_dict_to_df(data_dict):
    """
    Recursively find the region information for all nested objects.
    """
    # Column names
    names = [
        ("id", np.float64),
        ("atlas_id", np.float64),
        ("ontology_id", np.float64),
        ("acronym", str),
        ("name", str),
        ("color_hex_triplet", str),
        ("graph_order", np.float64),
        ("st_level", np.float64),
        ("hemisphere_id", np.float64),
        ("parent_structure_id", np.float64),
    ]
    # Making regions ID dataframe
    df = pd.DataFrame(columns=[i[0] for i in names])
    # Adding current region info to df
    df = pd.concat(
        [
            df,
            pd.DataFrame([data_dict[i[0]] for i in names], index=df.columns).T,
        ],
        axis=0,
        ignore_index=True,
    )
    # Recursively get the region information for all children
    for i in data_dict["children"]:
        df = pd.concat(
            [
                df,
                nested_tree_dict_to_df(i),
            ],
            axis=0,
            ignore_index=True,
        )
    # Casting columns to given types
    for i in names:
        df[i[0]] = df[i[0]].astype(i[1])
    # Returning the region info df
    return df
