import os
import subprocess
from enum import Enum
from optparse import NO_DEFAULT
from types import UnionType
from typing import Callable, Optional, Union, get_args, get_origin

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)

# DEPTH = 10
DEPTH = 50

ROWS_PARTITION = 10000000


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


class RefVersions(Enum):
    AVERAGE_TEMPLATE_25 = "average_template_25"
    ARA_NISSL_25 = "ara_nissl_25"


class AnnotVersions(Enum):
    CCF_2017_25 = "ccf_2017_25"
    CCF_2016_25 = "ccf_2016_25"
    CCF_2015_25 = "ccf_2015_25"


class MapVersions(Enum):
    ABA_ANNOTATIONS = "ABA_annotations"
    CM_ANNOTATIONS = "CM_annotations"


NO_DEFAULT = "no_default"


class ConfigParamsModel(BaseModel):
    """
    Pydantic model for registration parameters.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # REFERENCE
    atlas_dir: str = "abcd"
    ref_v: RefVersions = RefVersions.AVERAGE_TEMPLATE_25
    annot_v: AnnotVersions = AnnotVersions.CCF_2016_25
    map_v: MapVersions = MapVersions.ABA_ANNOTATIONS
    # RAW
    chunksize: tuple[int, int, int] = PROC_CHUNKS
    # REGISTRATION
    ref_orient_ls: tuple[int, int, int] = (1, 2, 3)
    ref_z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    z_rough: int = 3
    y_rough: int = 6
    x_rough: int = 6
    z_fine: float = 1.0
    y_fine: float = 0.6
    x_fine: float = 0.6
    z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    # MASK
    mask_gaus_blur: int = 1
    mask_thresh: int = 300
    # OVERLAP
    depth: int = DEPTH
    # CELL COUNTING
    tophat_sigma: int = 10
    dog_sigma1: int = 1
    dog_sigma2: int = 4
    gauss_sigma: int = 101
    thresh_p: int = 60
    min_threshd: int = 100
    max_threshd: int = 10000
    maxima_sigma: int = 10
    min_wshed: int = 1
    max_wshed: int = 1000


def enum2list(my_enum):
    return [e.value for e in my_enum]


# from microscopy_proc.constants import RefVersions
# from microscopy_proc.utils.misc_utils import enum2list

# TODO: implement none and default options


class ConfigsUpdater:
    """
    There's containers and columns, but returns
    the inputted values in the end
    (i.e. containers not returned).
    """

    @staticmethod
    def init_inputter(
        label: str,
        curr=None,
        nullable: bool = False,
        default=NO_DEFAULT,
        container=None,
    ):
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # container = container.container(border=True)
        # If nullable, making nullable checkbox
        is_null = False
        if nullable:
            is_null = container.toggle(
                "Set a None",
                value=curr is None,
                key=f"{label}_null",
            )
        # If default is set, making default button
        if default != NO_DEFAULT:
            default_clicked = container.button(
                label=f"Set as default (`{default}`)",
                disabled=is_null,
                key=f"{label}_default",
            )
            # If button clicked, then setting curr to default
            if default_clicked:
                curr = default
        return container, curr, is_null

    @staticmethod
    def enum_input(
        label: str,
        my_enum,
        curr=None,
        nullable: bool = False,
        default=NO_DEFAULT,
        container=None,
    ):
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label,
            curr,
            nullable,
            default,
            container,
        )
        # Selectbox
        output = container.selectbox(
            label=label,
            options=enum2list(my_enum),
            index=enum2list(my_enum).index(curr.value) if curr else None,
            disabled=is_null,
            key=label,
            label_visibility="hidden",
        )
        # Returning input
        return None if is_null else my_enum(output)

    @staticmethod
    def int_input(
        label: str,
        curr=None,
        nullable=False,
        default=NO_DEFAULT,
        container=None,
    ):
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label,
            curr,
            nullable,
            default,
            container,
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=1,
            key=label,
            label_visibility="collapsed",
        )  # type: ignore
        # Returning input
        return None if is_null else output

    @staticmethod
    def float_input(
        label: str,
        curr=None,
        nullable=False,
        default=NO_DEFAULT,
        container=None,
    ):
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label,
            curr,
            nullable,
            default,
            container,
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=0.05,
            key=label,
            label_visibility="collapsed",
        )  # type: ignore
        # Returning input
        return None if is_null else output

    @staticmethod
    def str_input(
        label: str,
        curr=None,
        nullable=False,
        default=NO_DEFAULT,
        container=None,
    ):
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label,
            curr,
            nullable,
            default,
            container,
        )
        output = st.text_input(
            label=label,
            value=default,
            key=label,
            label_visibility="collapsed",
        )
        return None if is_null else output

    @staticmethod
    def tuple_inputs(
        label: str,
        n: int,
        func: Callable | tuple[Callable, ...],
        nullable: bool | tuple[bool, ...] = False,
        default=NO_DEFAULT,
        container=None,
        n_labels: Optional[tuple[str, ...]] = None,
        **kwargs,
        # is_int=True,
        # curr=None,
        # n_labels=None,
    ):
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # For each number input, making into list to input to func
        columns_ls = container.columns(n)
        output_ls = [None for _ in range(n)]
        func_ls = func if isinstance(func, tuple) else [func for _ in range(n)]
        nullable_ls = (
            nullable if isinstance(nullable, tuple) else [nullable for _ in range(n)]
        )
        default_ls = (
            default if isinstance(default, tuple) else [default for _ in range(n)]
        )
        n_labels_ls = (
            n_labels if isinstance(n_labels, tuple) else [str(i) for i in range(n)]
        )
        n_labels_ls = [f"{label} - {n_label}" for n_label in n_labels_ls]
        # Making kwargs into kwargs_ls dict of lists
        kwargs_ls = {
            k: v if isinstance(v, tuple) else [v for _ in range(n)]
            for k, v in kwargs.items()
        }
        # Asserting all kwargs_ls elements are equal to n
        for k, v in kwargs_ls.items():
            assert len(v) == n
        # "Transposing" kwargs_ls so it becomes a list of dicts.
        kwargs_ls = [{k: v[i] for k, v in kwargs_ls.items()} for i in range(n)]
        # Making inputs
        output_ls = tuple(
            func_ls[i](
                label=n_labels_ls[i],
                nullable=nullable_ls[i],
                default=default_ls[i],
                container=columns_ls[i],
                **kwargs_ls[i],
            )
            for i in range(n)
        )
        # Returning input
        return output_ls

    @classmethod
    def type2updater(cls, my_type: type) -> Callable:
        if issubclass(my_type, Enum):
            return lambda label, curr, nullable, default, container: cls.enum_input(
                label=label,
                my_enum=my_type,
                curr=curr,
                nullable=nullable,
                default=default,
                container=container,
            )
        elif my_type is int:
            return cls.int_input
        elif my_type is float:
            return cls.float_input
        elif my_type is str:
            return cls.str_input
        else:
            raise NotImplementedError(f"Type {my_type} not implemented")

    @staticmethod
    def check_type_n_nullable(my_type):
        """
        Returns tuple of:
        - The non-nullable type
        - Whether the type is nullable
        """
        origin = get_origin(my_type) or my_type
        args = list(get_args(my_type))
        nullable = False
        # If a Union, then checking if None is possible type
        if origin in [Union, UnionType, Optional]:
            if None in args:
                args.remove(None)
                nullable = True
            # Returns (non-nullable type, is_nullable)
            # TODO: what happens if multiple args?
            return nullable, args[0]
        # If not a Union, then returning the origin type
        return nullable, origin

    @classmethod
    def field2updater(
        cls,
        pydantic_instance: BaseModel,
        field_name: str,
        n_labels: Optional[tuple[str, ...]] = None,
        container=None,
        **kwargs,
    ):
        """
        Recursively builds a streamlit widget for a pydantic field.
        """
        # Getting field value and info
        curr = getattr(pydantic_instance, field_name)
        field_info = pydantic_instance.model_fields[field_name]
        default = field_info.default
        # Getting type and nullable
        nullable, my_type = cls.check_type_n_nullable(field_info.annotation)
        args = get_args(field_info.annotation)

        # If a tuple (e.g. tuple)
        if issubclass(my_type, tuple):
            # Building tuple inputs
            funcs_ls = []
            nullable_ls = []
            n_labels = n_labels or tuple(str(i) for i in range(len(curr)))
            for i, arg in enumerate(args):
                nullable, my_type = cls.check_type_n_nullable(arg)
                funcs_ls.append(cls.type2updater(my_type))
                nullable_ls.append(nullable)
            return cls.tuple_inputs(
                label=field_name,
                n=len(curr),
                func=tuple(funcs_ls),
                nullable=tuple(nullable_ls),
                default=default,
                container=container,
                n_labels=n_labels,
                curr=curr,
                **kwargs,
            )
        # Otherwise, using type2updated immediately
        return cls.type2updater(my_type)(
            label=field_name,
            curr=curr,
            nullable=nullable,
            default=default,
            container=container,
            **kwargs,
        )


#####################################################################
# Pipeline Functions (callbacks)
#####################################################################

if "configs" not in st.session_state:
    st.session_state["configs"] = ConfigParamsModel()


def init_proj(proj_dir: str):
    # TODO: check that proj_dir is valid
    st.session_state["proj_dir"] = proj_dir
    st.success("Project Initialised")


def load_curr_proj():
    # Recalling session state variables
    proj_dir = st.session_state.get("proj_dir", None)
    # Showing project description in sidebar
    with st.sidebar:
        st.subheader(f"Root Directory: {proj_dir}")
        # Placeholder
        st.write("No project initialised")
        st.dataframe(
            pd.DataFrame(
                np.random.randn(5, 5),
            )
        )


# #####################################################################
# # Streamlit pages
# #####################################################################


def page_init_proj():
    load_curr_proj()
    # Select project
    st.write("## Init Project")
    proj_dir = st.text_input(
        label="Root Directory",
        value=st.session_state.get("proj_dir", "/"),
        key="proj_dir_input",
    )
    st.button(
        "Confirm project directory",
        on_click=init_proj,
        args=(proj_dir,),
    )
    # TODO: check if project directory is valid
    # TODO: allow create new project


def page_configs():
    load_curr_proj()
    st.write("# Project Configs")
    # proj_dir = st.session_state.get("proj_dir", None)
    # pfm = get_proj_fp_model(proj_dir)
    # configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
    configs: ConfigParamsModel = st.session_state.get("configs", None)

    with st.expander("See Configs"):
        # JSON configs
        st.json(configs.model_dump())

    st.header("Edit Configs")
    with st.expander("Reference"):
        configs.atlas_dir = ConfigsUpdater.field2updater(configs, "atlas_dir")
        configs.ref_v = ConfigsUpdater.field2updater(configs, "ref_v")
        configs.annot_v = ConfigsUpdater.field2updater(configs, "annot_v")
        configs.map_v = ConfigsUpdater.field2updater(configs, "map_v")
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
        configs.z_rough = ConfigsUpdater.field2updater(configs, "z_rough")
        configs.y_rough = ConfigsUpdater.field2updater(configs, "y_rough")
        configs.x_rough = ConfigsUpdater.field2updater(configs, "x_rough")
        configs.z_fine = ConfigsUpdater.field2updater(configs, "z_fine")
        configs.y_fine = ConfigsUpdater.field2updater(configs, "y_fine")
        configs.x_fine = ConfigsUpdater.field2updater(configs, "x_fine")
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
        configs.mask_gaus_blur = ConfigsUpdater.field2updater(configs, "mask_gaus_blur")
        configs.mask_thresh = ConfigsUpdater.field2updater(configs, "mask_thresh")
    with st.expander("Overlap"):
        configs.depth = ConfigsUpdater.field2updater(configs, "depth")
    with st.expander("Cell Counting"):
        configs.tophat_sigma = ConfigsUpdater.field2updater(configs, "tophat_sigma")
        configs.dog_sigma1 = ConfigsUpdater.field2updater(configs, "dog_sigma1")
        configs.dog_sigma2 = ConfigsUpdater.field2updater(configs, "dog_sigma2")
        configs.gauss_sigma = ConfigsUpdater.field2updater(configs, "gauss_sigma")
        configs.thresh_p = ConfigsUpdater.field2updater(configs, "thresh_p")
        configs.min_threshd = ConfigsUpdater.field2updater(configs, "min_threshd")
        configs.max_threshd = ConfigsUpdater.field2updater(configs, "max_threshd")
        configs.maxima_sigma = ConfigsUpdater.field2updater(configs, "maxima_sigma")
        configs.min_wshed = ConfigsUpdater.field2updater(configs, "min_wshed")
        configs.max_wshed = ConfigsUpdater.field2updater(configs, "max_wshed")

    # Button: Save
    # Button: Revert
    columns = st.columns(2, vertical_alignment="center")
    columns[0].button(
        label="Reset",
        key="configs_reset",
        # on_click=lambda: ConfigParamsModel.model_validate(read_json(pfm.config_params)),
        # TODO: read configs
    )
    columns[1].button(
        label="Save",
        key="configs_save",
        # on_click=lambda: write_json(fp, configs.model_dump()),
    )


def page_placeholder2():
    st.write("## Placeholder page2 ")


#####################################################################
# Streamlit application
#####################################################################


def main():
    st.title("Microscopy Processing Pipeline")

    pg = st.navigation(
        [
            st.Page(page_init_proj, title="Init Project"),
            st.Page(page_configs, title="Placeholder"),
            st.Page(page_placeholder2, title="Placeholder2"),
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
