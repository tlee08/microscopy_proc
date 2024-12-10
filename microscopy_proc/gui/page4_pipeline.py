import streamlit as st

from microscopy_proc.pipeline_funcs.pipeline_funcs import PipelineFuncs
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import PROJ_DIR, init_var, page_decorator

PIPELINE = "pipeline"
CHECKBOXES = f"{PIPELINE}_checkboxes"
OVERWRITE = f"{PIPELINE}_overwrite"
RUN = f"{PIPELINE}_run"


@page_decorator()
def page4_pipeline():
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
    init_var(
        CHECKBOXES,
        # TODO: dynamically generate this dictionary
        {
            PipelineFuncs.ref_prepare: False,
            PipelineFuncs.img_rough: False,
            PipelineFuncs.img_fine: False,
            PipelineFuncs.img_trim: False,
            PipelineFuncs.elastix_registration: False,
            PipelineFuncs.make_mask: False,
            PipelineFuncs.img_overlap: False,
            PipelineFuncs.cellc1: False,
            PipelineFuncs.cellc2: False,
            PipelineFuncs.cellc3: False,
            PipelineFuncs.cellc4: False,
            PipelineFuncs.cellc5: False,
            PipelineFuncs.cellc6: False,
            PipelineFuncs.cellc7: False,
            PipelineFuncs.cellc8: False,
            PipelineFuncs.cellc9: False,
            PipelineFuncs.cellc10: False,
            PipelineFuncs.cellc11: False,
            PipelineFuncs.cellc_coords_only: False,
            PipelineFuncs.transform_coords: False,
            PipelineFuncs.cell_mapping: False,
            PipelineFuncs.group_cells: False,
            PipelineFuncs.cells2csv: False,
            PipelineFuncs.coords2points_raw: False,
            PipelineFuncs.coords2heatmap_raw: False,
            PipelineFuncs.coords2points_trfm: False,
            PipelineFuncs.coords2heatmap_trfm: False,
        },
    )
    init_var(OVERWRITE, False)

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
        st.checkbox(
            label=func.__name__,
            value=pipeline_checkboxes[func],
            key=f"{PIPELINE}_{func.__name__}",
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
                func(
                    pfm=pfm,
                    overwrite=st.session_state[OVERWRITE],
                )
