import os
import subprocess

import streamlit as st

from microscopy_proc.scripts.gui_funcs import (
    ConfigsUpdater,
    init_session_state,
    load_configs,
    make_proj_func,
    page_decorator,
    save_configs,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

# from microscopy_proc.scripts.gui_funcs import ConfigsUpdater, enum2list


# #####################################################################
# # Streamlit pages
# #####################################################################


@page_decorator(check_proj_dir=False)
def page_init_proj():
    # Title
    st.write("## Init Project")
    # Input: Root Directory
    proj_dir = st.text_input(
        label="Root Directory",
        value=st.session_state.get("proj_dir", "/"),
        key="proj_dir_input",
    )
    # Button: Confirm
    proj_dir_confirm = st.button(
        label="Confirm project directory",
        key="proj_dir_confirm",
    )
    if proj_dir_confirm:
        # Checking if project directory exists
        if not os.path.exists(proj_dir):
            st.error("Project directory does not exist")
        else:
            # Storing project directory
            st.session_state["proj_dir"] = proj_dir
            try:
                # Project directory is initialised (has configs file)
                load_configs()
                st.success("Loaded project directory")
            except FileNotFoundError:
                # Project directory is not initialised (give option to create)
                st.warning(
                    "Project directory does not contain config_params file.\n"
                    + "You can create a new project in this directory."
                )
                proj_dir_create = st.button(
                    label="Create new project",
                    on_click=make_proj_func,
                    args=(proj_dir,),
                    key="proj_dir_create",
                )


@page_decorator()
def page_configs():
    # Recalling session state variables
    configs: ConfigParamsModel = st.session_state.get("configs", None)

    st.write("# Edit Configs")
    with st.expander("Reference"):
        configs.atlas_dir = ConfigsUpdater.field2updater(configs, "atlas_dir")  # type: ignore
        configs.ref_v = ConfigsUpdater.field2updater(configs, "ref_v")  # type: ignore
        configs.annot_v = ConfigsUpdater.field2updater(configs, "annot_v")  # type: ignore
        configs.map_v = ConfigsUpdater.field2updater(configs, "map_v")  # type: ignore
    with st.expander("Raw"):
        configs.chunksize = ConfigsUpdater.field2updater(
            configs, "chunksize", ("z", "y", "x")
        )
    with st.expander("Registration"):
        configs.ref_orient_ls = ConfigsUpdater.field2updater(
            configs, "ref_orient_ls", ("z", "y", "x")
        )
        configs.ref_z_trim = ConfigsUpdater.field2updater(
            configs, "ref_z_trim", ("start", "stop", "step")
        )
        configs.ref_y_trim = ConfigsUpdater.field2updater(
            configs, "ref_y_trim", ("start", "stop", "step")
        )
        configs.ref_x_trim = ConfigsUpdater.field2updater(
            configs, "ref_x_trim", ("start", "stop", "step")
        )
        configs.z_rough = ConfigsUpdater.field2updater(configs, "z_rough")  # type: ignore
        configs.y_rough = ConfigsUpdater.field2updater(configs, "y_rough")  # type: ignore
        configs.x_rough = ConfigsUpdater.field2updater(configs, "x_rough")  # type: ignore
        configs.z_fine = ConfigsUpdater.field2updater(configs, "z_fine")  # type: ignore
        configs.y_fine = ConfigsUpdater.field2updater(configs, "y_fine")  # type: ignore
        configs.x_fine = ConfigsUpdater.field2updater(configs, "x_fine")  # type: ignore
        configs.z_trim = ConfigsUpdater.field2updater(
            configs, "z_trim", ("start", "stop", "step")
        )
        configs.y_trim = ConfigsUpdater.field2updater(
            configs, "y_trim", ("start", "stop", "step")
        )
        configs.x_trim = ConfigsUpdater.field2updater(
            configs, "x_trim", ("start", "stop", "step")
        )
    with st.expander("Mask"):
        configs.mask_gaus_blur = ConfigsUpdater.field2updater(configs, "mask_gaus_blur")  # type: ignore
        configs.mask_thresh = ConfigsUpdater.field2updater(configs, "mask_thresh")  # type: ignore
    with st.expander("Overlap"):
        configs.depth = ConfigsUpdater.field2updater(configs, "depth")  # type: ignore
    with st.expander("Cell Counting"):
        configs.tophat_sigma = ConfigsUpdater.field2updater(configs, "tophat_sigma")  # type: ignore
        configs.dog_sigma1 = ConfigsUpdater.field2updater(configs, "dog_sigma1")  # type: ignore
        configs.dog_sigma2 = ConfigsUpdater.field2updater(configs, "dog_sigma2")  # type: ignore
        configs.gauss_sigma = ConfigsUpdater.field2updater(configs, "gauss_sigma")  # type: ignore
        configs.thresh_p = ConfigsUpdater.field2updater(configs, "thresh_p")  # type: ignore
        configs.min_threshd = ConfigsUpdater.field2updater(configs, "min_threshd")  # type: ignore
        configs.max_threshd = ConfigsUpdater.field2updater(configs, "max_threshd")  # type: ignore
        configs.maxima_sigma = ConfigsUpdater.field2updater(configs, "maxima_sigma")  # type: ignore
        configs.min_wshed = ConfigsUpdater.field2updater(configs, "min_wshed")  # type: ignore
        configs.max_wshed = ConfigsUpdater.field2updater(configs, "max_wshed")  # type: ignore
    # Checking configs and updating in session_state
    configs = ConfigParamsModel.model_validate(configs)
    st.session_state["configs"] = configs

    # Showing updated configs
    st.write("# See Project Configs")
    with st.expander("See Configs"):
        # JSON configs
        st.json(configs.model_dump())

    # Button: Save
    st.button(label="Save", key="configs_save", on_click=save_configs)


@page_decorator()
def page_pipeline():
    # Recalling session state variables
    proj_dir = st.session_state["proj_dir"]
    pfm = get_proj_fp_model(proj_dir)

    st.write("## Pipeline")
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state["pipeline_overwrite"],
        key="pipeline_overwrite",
    )

    # Making pipeline checkboxes
    pipeline_checkboxes = st.session_state["pipeline_checkboxes"]
    for func in pipeline_checkboxes:
        pipeline_checkboxes[func] = st.checkbox(
            label=func.__name__,
            value=pipeline_checkboxes[func],
            key=func.__name__,
        )
    # Button to run pipeline
    st.button(
        label="Run pipeline",
        key="pipeline_run_btn",
    )
    if st.session_state["pipeline_run_btn"]:
        st.write("Running:")
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                st.write(f"- {func.__name__}")
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                func(pfm, overwrite=st.session_state["pipeline_overwrite"])


@page_decorator()
def page_visualiser():
    st.write("## Visualiser")


#####################################################################
# Streamlit application
#####################################################################


def main():
    # Initialising session state
    init_session_state()
    # Title
    st.title("Microscopy Processing Pipeline")
    # Multi-page navigation
    pg = st.navigation(
        [
            st.Page(page_init_proj),
            st.Page(page_configs),
            st.Page(page_pipeline),
            st.Page(page_visualiser),
            # st.Page(page_init_project, title="init_project"),
            # st.Page(page_update_configs, title="update_configs"),
            # st.Page(page_run_dlc, title="run_dlc"),
            # st.Page(page_calculate_params, title="calculate_params"),
            # st.Page(page_preprocess, title="preprocess"),
            # st.Page(page_extract_features, title="extract_features"),
            # st.Page(page_classify_behaviours, title="classify_behaviours"),
        ]
    )

    pg.run()


def run_script():
    """
    Running the streamlit script.

    Note that it must be run in a subprocess to make the call:
    ```
    streamlit run /path/to/gui.py
    ```
    """
    curr_fp = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", curr_fp])


if __name__ == "__main__":
    main()
