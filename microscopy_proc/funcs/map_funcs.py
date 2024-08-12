import numpy as np
import pandas as pd

from microscopy_proc.constants import CELL_MEASURES


def nested_tree_dict_to_df(data_dict: dict):
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


def combine_nested_regions(cells_grouped: pd.DataFrame, annot_df: pd.DataFrame):
    """
    Combine (sum) children regions in their parent regions in the cells_grouped dataframe.

    Done recursively.
    """
    # Storing the parent region names in `annot_df`
    annot_df = (
        pd.merge(
            left=annot_df,
            right=annot_df[["id", "acronym"]].rename(
                columns={"id": "parent_id", "acronym": "parent_acronym"}
            ),
            left_on="parent_structure_id",
            right_on="parent_id",
            how="left",
        )
        .drop(columns=["parent_id"])
        .set_index("id")
    )[["name", "acronym", "color_hex_triplet", "parent_structure_id", "parent_acronym"]]
    # Merging the cells_grouped df with the annot_df
    cells_grouped = pd.merge(
        left=annot_df,
        right=cells_grouped,
        left_index=True,
        right_index=True,
        how="outer",
    )
    # Making a children list column in cells_grouped
    cells_grouped["children"] = [[] for i in range(cells_grouped.shape[0])]
    for i in cells_grouped.index:
        i_parent = cells_grouped.loc[i, "parent_structure_id"]
        if not np.isnan(i_parent):
            cells_grouped.loc[i_parent, "children"].append(i)

    # Summing the cell count and volume for each region
    def r(i):
        # BASE CASE: no children - use current values
        # REC CASE: has children - recursively sum children values + current values
        cells_grouped.loc[i, cols] += np.sum(
            [r(j) for j in cells_grouped.loc[i, "children"]], axis=0
        )
        return cells_grouped.loc[i, cols]

    # Start from each root (i.e. nodes with no parent region)
    cols = list(CELL_MEASURES.values())
    cells_grouped[cols] = cells_grouped[cols].fillna(0)
    [r(i) for i in cells_grouped[cells_grouped["parent_structure_id"].isna()].index]
    # Removing unnecessary columns
    cells_grouped = cells_grouped.drop(columns=["children"])
    # Returning
    return cells_grouped


def df_to_nested_tree_dict(df: pd.DataFrame) -> dict:
    # Adding children list to each region
    df = df.copy()
    df["children"] = [[] for i in range(df.shape[0])]
    for i in df.index:
        i_parent = df.loc[i, "parent_structure_id"]
        if np.isnan(i_parent):
            continue
        if i_parent is None:
            df.loc[i_parent, "children"] = []
        df.loc[i_parent, "children"].append(i)

    # Converting to dict
    def r(i):
        # Storing info of current region in dict
        tree = df.loc[i].to_dict()
        # BASE CASE: no children
        if df.loc[i, "children"] == []:
            pass
        # REC CASE: has children - recursively get children info
        else:
            tree["children"] = [r(j) for j in df.loc[i, "children"]]
        return tree

    tree = r(df[df["parent_structure_id"].isna()].index[0], {})
    # Returning
    return tree
