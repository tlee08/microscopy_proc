import numpy as np
import pandas as pd


def nested_tree_dict2df(data_dict: dict):
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
                nested_tree_dict2df(i),
            ],
            axis=0,
            ignore_index=True,
        )
    # Casting columns to given types
    for i in names:
        df[i[0]] = df[i[0]].astype(i[1])
    # Returning the region info df
    return df


def combine_nested_regions(cells_agg_df: pd.DataFrame, annot_df: pd.DataFrame):
    """
    Combine (sum) children regions in their parent regions in the cells_agg dataframe.

    Done recursively.

    Notes
    -----
    - The `annot_df` is the annotation mappings dataframe.
    - The `cells_agg` is the cells dataframe grouped by region ID (so ID is the index).
    """
    # Getting the sum column names (i.e. all columns in cells_agg_d)
    sum_cols = cells_agg_df.columns
    # For each region, storing the parent region name in `annot_df`
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
        # .drop(columns=["parent_id"])
        .set_index("id")
    )[["name", "acronym", "color_hex_triplet", "parent_structure_id", "parent_acronym"]]
    # Merging the cells_agg df with the annot_df
    # NOTE: we are setting the annot_df index as ID
    # and assuming cells_agg index is ID (via groupby)
    cells_agg_df = pd.merge(
        left=annot_df,
        right=cells_agg_df,
        left_index=True,
        right_index=True,
        how="outer",
    )
    # Making a children list column in cells_agg
    cells_agg_df["children"] = [[] for i in range(cells_agg_df.shape[0])]
    for i in cells_agg_df.index:
        i_parent = cells_agg_df.loc[i, "parent_structure_id"]
        if not np.isnan(i_parent):
            cells_agg_df.loc[i_parent, "children"].append(i)

    # Summing the cell count and volume for each region
    def r(i):
        # BASE CASE: no children - use current values
        # REC CASE: has children - recursively sum children values + current values
        cells_agg_df.loc[i, sum_cols] += np.sum(
            [r(j) for j in cells_agg_df.loc[i, "children"]], axis=0
        )
        return cells_agg_df.loc[i, sum_cols]

    # Start from each root (i.e. nodes with no parent region)
    cells_agg_df[sum_cols] = cells_agg_df[sum_cols].fillna(0)
    [r(i) for i in cells_agg_df[cells_agg_df["parent_structure_id"].isna()].index]
    # Removing unnecessary columns
    cells_agg_df = cells_agg_df.drop(columns=["children"])
    # Returning
    return cells_agg_df


def df2nested_tree_dict(df: pd.DataFrame) -> dict:
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


def df_map_ids(cells_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    # Getting the annotation name for every cell (zyx coord)
    # Left-joining the cells dataframe with the annotation mappings dataframe
    cells_df = pd.merge(
        left=cells_df,
        right=annot_df,
        how="left",
        on="id",
    )
    # Setting points with ID == -1 as "invalid" label
    cells_df.loc[cells_df["id"] == -1, "name"] = "invalid"
    # Setting points with ID == 0 as "universe" label
    cells_df.loc[cells_df["id"] == 0, "name"] = "universe"
    # Setting points with no region map name (but have a positive ID value) as "no label" label
    cells_df.loc[cells_df["name"].isna(), "name"] = "no label"
    # Returning
    return cells_df
