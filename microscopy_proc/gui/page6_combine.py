import os

import streamlit as st
from natsort import natsorted

from microscopy_proc.gui.page1_init import INPUT_M as PDIR_INPUT_M
from microscopy_proc.pipelines.batch_combine import combine_ls_pipeline
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import ProjDirStatus, page_decorator

COMBINE = "combine"
USE_PDIR = f"{COMBINE}_use_pdir"
OVERWRITE = f"{COMBINE}_overwrite"
INPUT_ROOT = f"{COMBINE}_input_root"
INPUT_ROOT_STATUS = f"{COMBINE}_input_root_status"
INPUT_OUT = f"{COMBINE}_input_out"
INPUT_OUT_STATUS = f"{COMBINE}_input_out_status"
CHECKBOXES = f"{COMBINE}_checkboxes"
DISABLED = f"{COMBINE}_disabled"
RUN = f"{COMBINE}_run"


def update_disabled():
    # Disabled is True if either input_root or input_out is not valid
    st.session_state[DISABLED] = (
        st.session_state[INPUT_ROOT_STATUS] != ProjDirStatus.VALID
        or st.session_state[INPUT_OUT_STATUS] != ProjDirStatus.VALID
    )


def use_pdir_func():
    # Updating session_state: INPUT_ROOT from PDIR_INPUT_M (in page1_init)
    st.session_state[INPUT_ROOT] = st.session_state.get(PDIR_INPUT_M, None)


def input_root_func():
    # Updating session state: INPUT_ROOT_STATUS
    if not st.session_state[INPUT_ROOT]:
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT_ROOT]):
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.VALID
    # Updating session state: DISABLED
    update_disabled()


def input_out_func():
    # Updating session state: INPUT_OUT_STATUS
    if not st.session_state[INPUT_OUT]:
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT_OUT]):
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.VALID
    # Updating session state: DISABLED
    update_disabled()


@page_decorator(check_proj_dir=False)
def page6_combine():
    # Initialising session state variables
    if COMBINE not in st.session_state:
        st.session_state[COMBINE] = True
        st.session_state[OVERWRITE] = False
        st.session_state[INPUT_ROOT] = None
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.NOT_SET
        st.session_state[INPUT_OUT] = None
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.NOT_SET
        st.session_state[CHECKBOXES] = []
        st.session_state[DISABLED] = True

    # Recalling session state variables

    # Title
    st.write("## Combine Projects in a Directory")
    # Button: use root directory from page1_init
    st.button(
        label="Use Root Directory from 'Init Project'",
        on_click=use_pdir_func,
        disabled=st.session_state[PDIR_INPUT_M] is None,
        key=USE_PDIR,
    )
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )
    # Input: Root Projects Directory
    st.text_input(
        label="Root Directory",
        value=st.session_state[INPUT_ROOT],
        on_change=input_root_func,
        key=INPUT_ROOT,
    )
    # Input: Output Directory
    st.text_input(
        label="Output Directory",
        value=st.session_state[INPUT_OUT],
        on_change=input_out_func,
        key=INPUT_OUT,
    )
    # Making checkboxes for projects inside root directory
    st.write("### Select Projects")
    if st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.NOT_SET:
        st.warning("Root directory not set")
    elif st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.NOT_EXIST:
        st.warning("Root directory does not exist")
    elif st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.VALID:
        with st.container(height=250):
            # Making checkboxes (only including ones with valid project dirs)
            st.session_state[CHECKBOXES] = []
            for i in natsorted(os.listdir(st.session_state[INPUT_ROOT])):
                # Checking current option has cells_agg and mask df files
                pdir_i = os.path.join(st.session_state[INPUT_ROOT], i)
                pfm_i = get_proj_fp_model(pdir_i)
                if os.path.isfile(pfm_i.cells_agg_df) and os.path.isfile(pfm_i.mask_df):
                    # Adding checkbox
                    st.session_state[CHECKBOXES].append(
                        st.checkbox(
                            label=i,
                            key=f"{COMBINE}_{i}",
                        )
                    )
    # Button: Combine projects
    st.button(
        label="Combine",
        disabled=st.session_state[DISABLED],
        key=RUN,
    )
    if st.session_state[RUN]:
        proj_dir_ls = [k for k, v in st.session_state[CHECKBOXES].items() if v]
        st.write("Combining:")
        for i in proj_dir_ls:
            st.write(f"- {i}")
        # Running combine func
        combine_ls_pipeline(
            proj_dir_ls,
            st.session_state[INPUT_OUT],
            overwrite=st.session_state[OVERWRITE],
        )
