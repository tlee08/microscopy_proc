import os

import streamlit as st

from microscopy_proc.pipelines.pipeline_funcs import tiff2zarr_pipeline
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import PROJ_DIR, ProjDirStatus, page_decorator

IMPORT = "import"
OVERWRITE = f"{IMPORT}_overwrite"
INPUT_SRC = f"{IMPORT}_input_src"
INPUT_SRC_STATUS = f"{IMPORT}_input_src_status"
DISABLED = f"{IMPORT}_disabled"
RUN = f"{IMPORT}_run"


def input_src_func():
    # Updating session state: INPUT_SRC_STATUS
    if not st.session_state[INPUT_SRC]:
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT_SRC]):
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.VALID
    # Updating session state: DISABLED
    st.session_state[DISABLED] = (
        st.session_state[INPUT_SRC_STATUS] != ProjDirStatus.VALID
    )


@page_decorator()
def page3_import():
    """ """
    # Initialising session state variables
    if IMPORT not in st.session_state:
        st.session_state[IMPORT] = True
        st.session_state[OVERWRITE] = False
        st.session_state[INPUT_SRC] = None
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.NOT_SET
        st.session_state[DISABLED] = True

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = get_proj_fp_model(proj_dir)

    st.write("## Import Image")
    st.write("Imports image into project directory as a zarr chunked file.")
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )
    # Input: Source file or directory of image
    st.text_input(
        label="Image source filepath",
        value=st.session_state[INPUT_SRC],
        on_change=input_src_func,
        key=INPUT_SRC,
    )
    # Button: Combine projects
    st.button(
        label="Run",
        disabled=st.session_state[DISABLED],
        key=RUN,
    )
    # Running pipeline
    if st.session_state[RUN]:
        st.write("Running")
        st.write(f"Importing image from {st.session_state[INPUT_SRC]} to {proj_dir}")
        # Running tiff2zarr
        tiff2zarr_pipeline(
            pfm=pfm,
            src_fp=st.session_state[INPUT_SRC],
            overwrite=st.session_state[OVERWRITE],
        )
