import os

import streamlit as st

from microscopy_proc.gui.gui_funcs import load_configs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    make_proj_dirs,
    update_configs,
)

from .gui_funcs import ProjDirStatus, page_decorator


def proj_dir_set_func():
    """
    Changes the project directory in session state.
    Runs relevant checks.
    """
    # Retrieving session state variables
    proj_dir = st.session_state["proj_dir_input"]
    # Setting session_state variables based on proj_dir checks
    # Checking if project directory exists
    if not os.path.exists(proj_dir):
        st.session_state["proj_dir_status"] = ProjDirStatus.NOT_EXIST
    else:
        # Storing project directory
        st.session_state["proj_dir"] = proj_dir
        try:
            # Project directory is initialised (has configs file)
            load_configs()
            st.session_state["proj_dir_status"] = ProjDirStatus.VALID
        except FileNotFoundError:
            # Project directory is not initialised (give option to create)
            st.session_state["proj_dir_status"] = ProjDirStatus.NOT_INIT


def proj_dir_create_func():
    """
    Function to make new project.

    Makes project folders and configs file.
    """
    proj_dir = st.session_state["proj_dir"]
    make_proj_dirs(proj_dir)
    pfm = get_proj_fp_model(proj_dir)
    update_configs(pfm)
    st.success("Created new project")
    # Rerunning set func to update session state
    proj_dir_set_func()


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
    # Title
    st.write("## Init Project")
    # tabs: single or multi project
    tabs = st.tabs(["Single Project", "Multiple Projects"])
    with tabs[0]:
        # Input: Project Directory
        st.text_input(
            label="Project Directory",
            value=st.session_state.get("proj_dir", "/"),
            key="proj_dir_input",
        )
    with tabs[1]:
        # Input: Root Projects Directory
        st.text_input(
            label="Root Directory",
            value=st.session_state.get("proj_dir", "/"),
            key="root_projs_dir_input",
        )
        # selectbox: folders (i.e. projects) inside root directory
        st.selectbox(
            label="Projects",
            options=os.listdir(st.session_state["root_projs_dir_input"]),
            index=st.session_state.get("proj_dir_select", 0),
        )
        # Setting project directory
        st.session_state["proj_dir"] = os.path.join(
            st.session_state["root_projs_dir_input"],
            st.session_state["proj_dir_select"],
        )

    # Button: Set project directory
    st.button(
        label="Set project directory",
        on_click=proj_dir_set_func,
        key="proj_dir_set",
    )
    # container: outcome of project directory input
    with st.container():
        if st.session_state["proj_dir_status"] == ProjDirStatus.NOT_SET:
            st.warning("Project directory not set")
        elif st.session_state["proj_dir_status"] == ProjDirStatus.NOT_EXIST:
            st.error(
                "Project directory does not exist.\n\n"
                + "Reverting to existing project directory (if one is set)."
            )
        elif st.session_state["proj_dir_status"] == ProjDirStatus.NOT_INIT:
            st.warning(
                "Project directory does not contain config_params file.\n\n"
                + "You can create a new project in this directory."
            )
            st.button(
                label="Create new project",
                on_click=proj_dir_create_func,
                key="proj_dir_create",
            )
        elif st.session_state["proj_dir_status"] == ProjDirStatus.VALID:
            st.success("Loaded project directory")
