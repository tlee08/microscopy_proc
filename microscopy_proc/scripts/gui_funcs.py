import functools
from enum import Enum
from optparse import NO_DEFAULT
from types import UnionType
from typing import Callable, Optional, Union, get_args, get_origin

import streamlit as st
from pydantic import BaseModel

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
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.io_utils import read_json, write_json
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

NO_DEFAULT = "no_default"


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
        # Initialising container, nullable, and default widgets
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label, curr, nullable, default, container
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
        # Initialising container, nullable, and default widgets
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label, curr, nullable, default, container
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
        # Initialising container, nullable, and default widgets
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label, curr, nullable, default, container
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
        # Initialising container, nullable, and default widgets
        container, curr, is_null = ConfigsUpdater.init_inputter(
            label, curr, nullable, default, container
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


def init_session_state():
    if "pipeline_checkboxes" not in st.session_state:
        st.session_state["pipeline_checkboxes"] = {
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
    if "pipeline_overwrite" not in st.session_state:
        st.session_state["pipeline_overwrite"] = False


def load_configs():
    """
    Loading in configs to session state from project directory.
    """
    if "proj_dir" in st.session_state:
        proj_dir = st.session_state["proj_dir"]
        pfm = get_proj_fp_model(proj_dir)
        fp = pfm.config_params
        st.session_state["configs"] = ConfigParamsModel.model_validate(read_json(fp))


def save_configs():
    """
    Saving configs from session state to project directory.
    """
    if "configs" in st.session_state:
        configs = st.session_state["configs"]
        proj_dir = st.session_state["proj_dir"]
        pfm = get_proj_fp_model(proj_dir)
        fp = pfm.config_params
        write_json(fp, configs.model_dump())


#####################################################################
# Common Funcs
#####################################################################


def page_default_setup():
    is_proj_exists = True
    # Recalling session state variables
    proj_dir = st.session_state.get("proj_dir", None)
    # Checking if project configs exists
    try:
        pfm = get_proj_fp_model(proj_dir) if proj_dir else None
        ConfigParamsModel.model_validate(read_json(pfm.config_params))
    except AttributeError:
        is_proj_exists = False
        st.error("Project directory not initialised")
    except FileNotFoundError:
        is_proj_exists = False
    # Showing project description in sidebar
    with st.sidebar:
        if is_proj_exists:
            st.write("Project directory is initialised")
        else:
            st.subheader(f"Root Directory: {proj_dir}")
        # Placeholder
        # st.dataframe(
        #     pd.DataFrame(
        #         np.random.randn(5, 5),
        #     )
        # )
    return is_proj_exists


def page_decorator(check_proj_dir=True):
    def decorator_wrapper(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            is_proj_exists = page_default_setup()
            if check_proj_dir and not is_proj_exists:
                return
            return f(*args, **kwargs)

        return wrapper

    return decorator_wrapper
