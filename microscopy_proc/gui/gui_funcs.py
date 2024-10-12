import functools
from enum import Enum

import streamlit as st

from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.io_utils import read_json, write_json
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)


class ProjDirStatus(Enum):
    NOT_SET = "not_set"
    NOT_EXIST = "not_exist"
    NOT_INIT = "not_init"
    VALID = "valid"


#####################################################################
# Pipeline Functions (callbacks)
#####################################################################


def load_configs():
    """
    Loading in configs to session state from project directory.

    NOTE: does not catch errors
    """
    proj_dir = st.session_state["proj_dir"]
    pfm = get_proj_fp_model(proj_dir)
    fp = pfm.config_params
    st.session_state["configs"] = ConfigParamsModel.model_validate(read_json(fp))


def save_configs():
    """
    Saving configs from session state to project directory.

    NOTE: does not catch errors
    """
    configs = st.session_state["configs"]
    proj_dir = st.session_state["proj_dir"]
    pfm = get_proj_fp_model(proj_dir)
    fp = pfm.config_params
    write_json(fp, configs.model_dump())


#####################################################################
# Common Funcs
#####################################################################


def page_decorator(check_proj_dir=True):
    def decorator_wrapper(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Recalling session state variables
            proj_dir = st.session_state["proj_dir"]
            proj_dir_status = st.session_state["proj_dir_status"]
            # Checking whether project exists
            with st.sidebar:
                st.subheader(f"Root Directory: {proj_dir}")
                # Outputting project directory status
                if proj_dir_status == ProjDirStatus.NOT_SET:
                    st.warning("Project directory is not set")
                elif proj_dir_status == ProjDirStatus.NOT_EXIST:
                    st.warning("Project directory does not exist")
                elif proj_dir_status == ProjDirStatus.NOT_INIT:
                    st.warning("Project directory is not initialised")
                elif proj_dir_status == ProjDirStatus.VALID:
                    load_configs()
                    st.success("Loaded project directory and configs")
            # If project not exists and check_proj_dir is True
            # then don't render rest of page
            if check_proj_dir and proj_dir_status != ProjDirStatus.VALID:
                st.error(
                    "Project is not initialised.\n\n"
                    + "Please set or create a project directory."
                )
                return
            return f(*args, **kwargs)

        return wrapper

    return decorator_wrapper
