import streamlit as st

from microscopy_proc.pipelines.pipeline_funcs import (
    cell_mapping_pipeline,
    cellc1_pipeline,
    cellc2_pipeline,
    cellc3_pipeline,
    cellc4_pipeline,
    cellc5_pipeline,
    cellc6_pipeline,
    cellc7_pipeline,
    cellc8_pipeline,
    cellc9_pipeline,
    cellc10_pipeline,
    cellc11_pipeline,
    cellc_coords_only_pipeline,
    cells2csv_pipeline,
    group_cells_pipeline,
    img_fine_pipeline,
    img_overlap_pipeline,
    img_rough_pipeline,
    img_trim_pipeline,
    make_mask_pipeline,
    ref_prepare_pipeline,
    registration_pipeline,
    tiff2zarr_pipeline,
    transform_coords_pipeline,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import PROJ_DIR, page_decorator

CHECKBOXES = "pipeline_checkboxes"
OVERWRITE = "pipeline_overwrite"
RUN = "pipeline_run"


@page_decorator()
def page_pipeline():
    """
    Displays the pipeline page in the GUI, allowing users to select and run various pipeline functions.

    This function performs the following tasks:
    1. Retrieves the project directory and project file model from the session state.
    2. Displays a toggle for overwriting existing data.
    3. Creates checkboxes for each pipeline function, allowing users to select which functions to run.
    4. Provides a button to run the selected pipeline functions.
    5. Executes the selected pipeline functions when the button is pressed, respecting the overwrite setting.

    Session State Variables:
    - proj_dir: The directory of the current project.
    - pipeline_overwrite: Boolean indicating whether to overwrite existing data.
    - pipeline_checkboxes: Dictionary mapping pipeline functions to their checkbox states.
    - pipeline_run_btn: Boolean indicating whether the "Run pipeline" button has been pressed.
    """
    # Initialising session state variables (if necessary)
    if CHECKBOXES not in st.session_state:
        st.session_state[CHECKBOXES] = {
            tiff2zarr_pipeline: False,
            ref_prepare_pipeline: False,
            img_rough_pipeline: False,
            img_fine_pipeline: False,
            img_trim_pipeline: False,
            registration_pipeline: False,
            make_mask_pipeline: False,
            img_overlap_pipeline: False,
            cellc1_pipeline: False,
            cellc2_pipeline: False,
            cellc3_pipeline: False,
            cellc4_pipeline: False,
            cellc5_pipeline: False,
            cellc6_pipeline: False,
            cellc7_pipeline: False,
            cellc8_pipeline: False,
            cellc9_pipeline: False,
            cellc10_pipeline: False,
            cellc11_pipeline: False,
            cellc_coords_only_pipeline: False,
            transform_coords_pipeline: False,
            cell_mapping_pipeline: False,
            group_cells_pipeline: False,
            cells2csv_pipeline: False,
        }
        st.session_state[OVERWRITE] = False

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = get_proj_fp_model(proj_dir)

    st.write("## Pipeline")
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )

    # Making pipeline checkboxes
    pipeline_checkboxes = st.session_state[CHECKBOXES]
    for func in pipeline_checkboxes:
        pipeline_checkboxes[func] = st.checkbox(
            label=func.__name__,
            value=pipeline_checkboxes[func],
            key=func.__name__,
        )
    # Button: run pipeline
    st.button(
        label="Run pipeline",
        key=RUN,
    )
    if st.session_state[RUN]:
        # Showing selected pipeline
        st.write("Running:")
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                st.write(f"- {func.__name__}")
        # TODO: ensure this is blocking
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                func(pfm, overwrite=st.session_state[OVERWRITE])
