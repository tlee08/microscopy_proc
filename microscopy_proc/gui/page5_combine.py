import os
from enum import Enum

import streamlit as st

from microscopy_proc.gui.page1_init import INPUT_M as PDIR_INPUT_M

from .gui_funcs import page_decorator

COMBINE = "combine"
COMBINE_STATUS = "combine_status"
USE_PDIR = "combine_use_pdir"
INPUT = "combine_input"
CHECKBOXES = "combine_checkboxes"
RUN = "combine_run"
DISABLED = "combine_disabled"


class CombineStatus(Enum):
    NOT_SET = "not_set"
    NOT_EXIST = "not_exists"
    VALID = "valid"


def use_pdir_func():
    # Updating session_state
    st.session_state[INPUT] = st.session_state.get(PDIR_INPUT_M, None)
    # Updating widgets
    st.session_state[f"{INPUT}_w"] = st.session_state[INPUT]


def input_func():
    # Updating session_state: INPUT
    st.session_state[INPUT] = st.session_state[f"{INPUT}_w"]
    # Updating session state: COMBINE_STATUS
    if not st.session_state[INPUT]:
        st.session_state[COMBINE_STATUS] = CombineStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT]):
        st.session_state[COMBINE_STATUS] = CombineStatus.NOT_EXIST
    else:
        st.session_state[COMBINE_STATUS] = CombineStatus.VALID
    # Updating session state: DISABLED
    st.session_state[DISABLED] = st.session_state[COMBINE_STATUS] != CombineStatus.VALID


def combine_run_func():
    # Placeholder for function
    pass


@page_decorator(check_proj_dir=False)
def page5_combine():
    # Initialising session state variables
    if INPUT not in st.session_state:
        st.session_state[COMBINE_STATUS] = CombineStatus.NOT_SET
        st.session_state[INPUT] = None
        st.session_state[CHECKBOXES] = []
        st.session_state[DISABLED] = True

    # Recalling session state variables

    # Title
    st.write("## Combine Projects in a Directory")
    # Button: use root directory from page1_init
    st.button(
        label="Use Root Directory from 'Init Project'",
        on_click=use_pdir_func,
        # disabled=st.session_state[PDIR_INPUT_M],
        key=USE_PDIR,
    )
    # Input: Root Projects Directory
    st.text_input(
        label="Root Directory",
        value=st.session_state[INPUT],
        on_change=input_func,
        key=f"{INPUT}_w",
    )
    # Making checkboxes for projects inside root directory
    st.write("### Select Projects")
    if st.session_state[COMBINE_STATUS] == CombineStatus.NOT_SET:
        st.warning("Root directory not set")
    elif st.session_state[COMBINE_STATUS] == CombineStatus.NOT_EXIST:
        st.warning("Root directory does not exist")
    elif st.session_state[COMBINE_STATUS] == CombineStatus.VALID:
        with st.container(height=250):
            st.session_state[CHECKBOXES] = {
                i: st.checkbox(
                    label=i,
                    key=f"{COMBINE}_{i}_w",
                )
                for i in os.listdir(st.session_state[INPUT])
            }
    # Button: Combine projects
    st.button(
        label="Combine",
        on_click=combine_run_func,
        disabled=st.session_state[DISABLED],
        key=RUN,
    )
    if st.session_state[RUN]:
        proj_ls = [k for k, v in st.session_state[CHECKBOXES].items() if v]
        st.write("Combining:")
        for i in proj_ls:
            st.write(f"- {i}")
        # Placeholder for function
