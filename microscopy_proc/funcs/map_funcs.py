import numpy as np
import pandas as pd

from microscopy_proc.constants import AnnotColumns, AnnotExtraColumns


def nested_tree_dict2df(data_dict: dict):
    """
    Recursively find the region information for all nested objects.
    """
    # Column names
    names = [
        (AnnotColumns.ID.value, np.float64),
        (AnnotColumns.ATLAS_ID.value, np.float64),
        (AnnotColumns.ONTOLOGY_ID.value, np.float64),
        (AnnotColumns.ACRONYM.value, str),
        (AnnotColumns.NAME.value, str),
        (AnnotColumns.COLOR_HEX_TRIPLET.value, str),
        (AnnotColumns.GRAPH_ORDER.value, np.float64),
        (AnnotColumns.ST_LEVEL.value, np.float64),
        (AnnotColumns.HEMISPHERE_ID.value, np.float64),
        (AnnotColumns.PARENT_STRUCTURE_ID.value, np.float64),
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
    for i in data_dict[AnnotExtraColumns.CHILDREN.value]:
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


def annot_df_get_parents(annot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the parent region information for all regions
    in the annotation mappings dataframe.

    Returns a new dataframe with index as region ID
    and the columns:
    - NAME
    - ACRONYM
    - COLOR_HEX_TRIPLET
    - PARENT_STRUCTURE_ID
    - PARENT_ACRONYM
    """
    # For each region (i.e. row), storing the parent region name in a column
    # by merging the annot_df on parent_structure_id
    # with the annot_df (as parent copy, so own id)
    annot_df = (
        pd.merge(
            left=annot_df,
            right=annot_df[[AnnotColumns.ID.value, AnnotColumns.ACRONYM.value]].rename(
                columns={
                    AnnotColumns.ID.value: AnnotExtraColumns.PARENT_ID.value,
                    AnnotColumns.ACRONYM.value: AnnotExtraColumns.PARENT_ACRONYM.value,
                }
            ),
            left_on=AnnotColumns.PARENT_STRUCTURE_ID.value,
            right_on=AnnotExtraColumns.PARENT_ID.value,
            how="left",
        ).set_index(AnnotColumns.ID.value)
    )[
        [
            AnnotColumns.NAME.value,
            AnnotColumns.ACRONYM.value,
            AnnotColumns.COLOR_HEX_TRIPLET.value,
            AnnotColumns.PARENT_STRUCTURE_ID.value,
            AnnotExtraColumns.PARENT_ACRONYM.value,
        ]
    ]
    return annot_df


def combine_nested_regions(cells_agg_df: pd.DataFrame, annot_df: pd.DataFrame):
    """
    Combine (sum) children regions in their parent regions in the cells_agg dataframe.

    Done recursively.

    Returns a new dataframe with index as region ID,
    the annotation columns, and the
    same columns as the input cells_agg dataframe.

    Notes
    -----
    - The `annot_df` is the annotation mappings dataframe.
    - The `cells_agg` is the cells dataframe grouped by region ID (so ID is the index).
    """
    # Getting the sum column names (i.e. all columns in cells_agg_d)
    sum_cols = cells_agg_df.columns
    # Getting df with parent region information for all regions
    annot_df = annot_df_get_parents(annot_df)
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
    cells_agg_df[AnnotExtraColumns.CHILDREN.value] = [
        [] for i in range(cells_agg_df.shape[0])
    ]
    # For each row (i.e. region), adding the current row ID to the parent's (by ID)
    # children column list
    for i in cells_agg_df.index:
        i_parent = cells_agg_df.loc[i, AnnotColumns.PARENT_STRUCTURE_ID.value]
        if not np.isnan(i_parent):
            cells_agg_df.loc[i_parent, AnnotExtraColumns.CHILDREN.value].append(i)

    # Recursively summing the cells_agg_df columns with each child's and current value
    def recursive_sum(i):
        # BASE CASE: no children - use current values
        # REC CASE: has children - recursively sum children values + current values
        cells_agg_df.loc[i, sum_cols] += np.sum(
            [
                recursive_sum(j)
                for j in cells_agg_df.loc[i, AnnotExtraColumns.CHILDREN.value]
            ],
            axis=0,
        )
        return cells_agg_df.loc[i, sum_cols]

    # Filling NaN values with 0
    cells_agg_df[sum_cols] = cells_agg_df[sum_cols].fillna(0)
    # For each root (i.e. nodes with no parent region), running recursive summing
    [
        recursive_sum(i)
        for i in cells_agg_df[
            cells_agg_df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()
        ].index
    ]
    # Removing unnecessary columns (AnnotExtraColumns.CHILDREN.value column)
    cells_agg_df = cells_agg_df.drop(columns=[AnnotExtraColumns.CHILDREN.value])
    # Returning
    return cells_agg_df


def df2nested_tree_dict(df: pd.DataFrame) -> dict:
    # Adding children list to each region
    df = df.copy()
    df[AnnotExtraColumns.CHILDREN.value] = [[] for i in range(df.shape[0])]
    for i in df.index:
        i_parent = df.loc[i, AnnotColumns.PARENT_STRUCTURE_ID.value]
        if np.isnan(i_parent):
            continue
        if i_parent is None:
            df.loc[i_parent, AnnotExtraColumns.CHILDREN.value] = []
        df.loc[i_parent, AnnotExtraColumns.CHILDREN.value].append(i)

    # Converting to dict
    def r(i):
        # Storing info of current region in dict
        tree = df.loc[i].to_dict()
        # BASE CASE: no children
        if df.loc[i, AnnotExtraColumns.CHILDREN.value] == []:
            pass
        # REC CASE: has children - recursively get children info
        else:
            tree[AnnotExtraColumns.CHILDREN.value] = [
                r(j) for j in df.loc[i, AnnotExtraColumns.CHILDREN.value]
            ]
        return tree

    tree = r(df[df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()].index[0], {})
    # Returning
    return tree


def df_map_ids(cells_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    # Getting the annotation name for every cell (zyx coord)
    # Left-joining the cells dataframe with the annotation mappings dataframe
    cells_df = pd.merge(
        left=cells_df,
        right=annot_df,
        how="left",
        on=AnnotColumns.ID.value,
    )
    # Setting points with ID == -1 as "invalid" label
    cells_df.loc[cells_df[AnnotColumns.ID.value] == -1, AnnotColumns.NAME.value] = (
        "invalid"
    )
    # Setting points with ID == 0 as "universe" label
    cells_df.loc[cells_df[AnnotColumns.ID.value] == 0, AnnotColumns.NAME.value] = (
        "universe"
    )
    # Setting points with no region map name (but have a positive ID value) as "no label" label
    cells_df.loc[cells_df[AnnotColumns.NAME.value].isna(), AnnotColumns.NAME.value] = (
        "no label"
    )
    # Returning
    return cells_df
