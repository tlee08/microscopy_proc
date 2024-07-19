import numpy as np
import pandas as pd


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
