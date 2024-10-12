import os

import streamlit as st

from microscopy_proc.gui.gui_funcs import PROJ_DIR, PROJ_DIR_STATUS, load_configs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    make_proj_dirs,
    update_configs,
)

from .gui_funcs import ProjDirStatus, page_decorator


def pdir_input_m_func():
    options = []
    if st.session_state["pdir_input_m"] is not None:
        if os.path.isdir(st.session_state["pdir_input_m"]):
            options = os.listdir(st.session_state["pdir_input_m"])
    # Setting pdir_select_options
    st.session_state["pdir_select_options"] = options


def pdir_set_func(proj_dir: str):
    """
    Changes the project directory in session state.
    Runs relevant checks.
    """
    # Setting session_state variables based on proj_dir checks
    # Checking if project directory exists
    if not os.path.exists(proj_dir):
        st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        # Storing project directory
        st.session_state[PROJ_DIR] = proj_dir
        try:
            # Project directory is initialised (has configs file)
            load_configs()
            st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.VALID
        except FileNotFoundError:
            # Project directory is not initialised (give option to create)
            st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_INIT


def pdir_create_func():
    """
    Function to make new project.

    Makes project folders and configs file.
    """
    proj_dir = st.session_state[PROJ_DIR]
    make_proj_dirs(proj_dir)
    pfm = get_proj_fp_model(proj_dir)
    update_configs(pfm)
    # Rerunning set func to update session state
    pdir_set_func(proj_dir)


@page_decorator(check_proj_dir=False)
def page_init_proj():
    """
    Initializes the project page in the GUI.
    This function sets up the user interface for initializing a project directory.
    It includes input fields for the root directory and buttons to confirm the
    project directory or create a new project if the directory is not initialized.

    The function performs the following steps:
    1. Displays the title "Init Project".
    2. Provides an input field for the user to specify the root directory.
    3. Includes a button to confirm the project directory.
    4. Checks if the specified project directory exists:
       - If it does not exist, displays an error message.
       - If it exists, attempts to load project configurations:
         - If configurations are found, displays a success message.
         - If configurations are not found, displays a warning and provides an
           option to create a new project in the specified directory.

    Note
    ----
    - The function uses Streamlit (`st`) for the GUI components.
    - The project directory and its state are managed using `st.session_state`.

    Raises
    -----
    - FileNotFoundError: If the project directory does not contain the required
      configuration files.
    """
    if "pdir_input_s" not in st.session_state:
        st.session_state["pdir_input_s"] = "/"
        st.session_state["pdir_input_m"] = "/"
        st.session_state["pdir_select"] = None
        st.session_state["pdir_select_options"] = []

    # Title
    st.write("## Init Project")
    # tabs: single or multi project
    tabs = st.tabs(["Single Project", "Multiple Projects"])
    with tabs[0]:
        # Input: Project Directory
        pdir_input_s = st.text_input(
            label="Project Directory",
            value=st.session_state["pdir_input_s"],
            key="pdir_input_s",
        )
        # Calculating disabled status
        disabled_single = pdir_input_s is None
        # Button: Set project directory using input
        st.button(
            label="Set project directory",
            on_click=pdir_set_func,
            args=(pdir_input_s,),
            disabled=disabled_single,
            key="pdir_set_s",
        )
    with tabs[1]:
        # Input: Root Projects Directory
        pdir_input_m = st.text_input(
            label="Root Directory",
            value=st.session_state["pdir_input_m"],
            on_change=pdir_input_m_func,
            key="pdir_input_m",
        )
        # selectbox: folders (i.e. projects) inside root directory
        options = st.session_state["pdir_select_options"]
        pdir_select = st.session_state["pdir_select"]
        index = options.index(pdir_select) if pdir_select in options else None
        pdir_select = st.selectbox(
            label="Projects",
            options=options,
            index=index,
            key="pdir_select",
        )
        # Calculating proj dir input path and disabled status
        pdir_input_comb_m = os.path.join(pdir_input_m or "", pdir_select or "")
        disabled_multi = pdir_input_m is None or pdir_select is False
        # Button: Set project directory using root and select
        st.button(
            label="Set project directory",
            on_click=pdir_set_func,
            args=(pdir_input_comb_m,),
            disabled=disabled_multi,
            key="pdir_set_m",
        )
    # container: outcome of project directory input
    with st.container():
        if st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_SET:
            st.warning("Project directory not set")
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_EXIST:
            st.error(
                "Project directory does not exist.\n\n"
                + "Reverting to existing project directory (if one is set)."
            )
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_INIT:
            st.warning(
                "Project directory does not contain config_params file.\n\n"
                + "You can create a new project in this directory."
            )
            st.button(
                label="Create new project",
                on_click=pdir_create_func,
                key="pdir_create",
            )
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.VALID:
            st.success("Project directory loaded")
