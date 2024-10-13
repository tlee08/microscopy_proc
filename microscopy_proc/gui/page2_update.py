from enum import Enum, EnumType
from types import NoneType, UnionType
from typing import Any, Callable, Optional, Union, get_args, get_origin

import streamlit as st
from pydantic import BaseModel
from streamlit.delta_generator import DeltaGenerator

from microscopy_proc.gui.gui_funcs import (
    CONFIGS,
    L_SLC,
    L_ZYX,
    PROJ_DIR,
    load_configs,
    page_decorator,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.io_utils import write_json
from microscopy_proc.utils.misc_utils import const2ls, dictlists2listdicts, enum2list
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

UPDATE = "update"
VALUE = "value"
IS_NONE = "is_none"
DEFAULT = "default"
CONFIGS_RESET = f"{UPDATE}_reset"
CONFIGS_SAVE = f"{UPDATE}_save"


class NO_DEFAULT:
    pass


SUBLABEL_MAP = {
    "chunksize": L_ZYX,
    "ref_orient_ls": L_ZYX,
    "ref_z_trim": L_SLC,
    "ref_y_trim": L_SLC,
    "ref_x_trim": L_SLC,
    "z_trim": L_SLC,
    "y_trim": L_SLC,
    "x_trim": L_SLC,
}


class ConfigsUpdater:
    """
    There's containers and columns, but returns
    the inputted values in the end
    (i.e. containers not returned).
    """

    @classmethod
    def init_inputter(
        cls,
        label: str,
        curr: Any = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
    ) -> tuple[DeltaGenerator, Any, bool]:
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # If default exists, making a default button
        if default != NO_DEFAULT:
            container.button(
                label=f"Set as default (`{default}`)",
                key=f"{UPDATE}_{label}_{DEFAULT}",
            )
            # If button clicked, setting curr to default
            # and update widgets accordingly
            if st.session_state[f"{UPDATE}_{label}_{DEFAULT}"]:
                curr = default
                st.session_state[f"{UPDATE}_{label}_{VALUE}"] = curr
                st.session_state[f"{UPDATE}_{label}_{IS_NONE}"] = curr is None
        # If nullable, making nullable checkbox
        is_none = False
        if nullable:
            is_none = container.toggle(
                label="Set to None",
                value=curr is None,
                key=f"{UPDATE}_{label}_{IS_NONE}",
            )
        # Returning container, current value, and whether that value is None
        return container, curr, is_none

    @classmethod
    def enum_input(
        cls,
        label: str,
        my_enum: EnumType,
        curr: Optional[Any] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[Any]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        # Selectbox
        output = container.selectbox(
            label=label,
            options=enum2list(my_enum),
            index=enum2list(my_enum).index(curr) if curr else None,
            disabled=is_none,
            key=f"{UPDATE}_{label}_{VALUE}",
            label_visibility="collapsed",
        )  # type: ignore
        # Returning input
        return None if is_none else output

    @classmethod
    def int_input(
        cls,
        label: str,
        curr: Optional[int] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[int]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=1,
            disabled=is_none,
            key=f"{UPDATE}_{label}_{VALUE}",
            label_visibility="collapsed",
        )  # type: ignore
        # Returning input
        return None if is_none else output

    @classmethod
    def float_input(
        cls,
        label: str,
        curr: Optional[float] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[float]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=0.05,
            disabled=is_none,
            key=f"{UPDATE}_{label}_{VALUE}",
            label_visibility="collapsed",
        )  # type: ignore
        # Returning input
        return None if is_none else output

    @classmethod
    def str_input(
        cls,
        label: str,
        curr: Optional[str] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[str]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.text_input(
            label=label,
            value=curr,
            disabled=is_none,
            key=f"{UPDATE}_{label}_{VALUE}",
            label_visibility="collapsed",
        )
        return None if is_none else output

    @classmethod
    def tuple_inputs(
        cls,
        label: str,
        n: int,
        func: Callable | tuple[Callable, ...],
        nullable: bool | tuple[bool, ...] = False,
        default: Any | tuple[Any, ...] = NO_DEFAULT,
        container=None,
        n_labels: Optional[tuple[str, ...]] = None,
        **kwargs,
    ) -> tuple[Any, ...]:
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # For each number input, making into list to input to func
        columns_ls = container.columns(n)
        output_ls = [None for _ in range(n)]
        func_ls = func if isinstance(func, tuple) else const2ls(func, n)
        nullable_ls = nullable if isinstance(nullable, tuple) else const2ls(nullable, n)
        default_ls = default if isinstance(default, tuple) else const2ls(default, n)
        n_labels_ls = n_labels if isinstance(n_labels, tuple) else range(n)
        n_labels_ls = [f"{label}_{n_label}" for n_label in n_labels_ls]
        # Making kwargs into kwargs_ls dict of lists
        kwargs_ls = {
            k: v if isinstance(v, tuple) else const2ls(v, n) for k, v in kwargs.items()
        }
        # Asserting all list lengths are equal to n
        assert len(func_ls) == n
        assert len(nullable_ls) == n
        assert len(default_ls) == n
        assert len(n_labels_ls) == n
        # Asserting all kwargs_ls list lengths are equal to n
        for k, v in kwargs_ls.items():
            assert len(v) == n
        # "Transposing" kwargs_ls so it becomes a list of dicts.
        kwargs_ls = dictlists2listdicts(kwargs_ls)
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
            return lambda **kwargs: cls.enum_input(my_enum=my_type, **kwargs)
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
        # If a Union, then checking if NoneType is a possible type
        if origin in [Union, UnionType, Optional] and NoneType in args:
            args.remove(NoneType)
            nullable = True
            # TODO: what happens if multiple args?
            origin = args[0]
        # Returns (non-nullable type, is_nullable)
        return nullable, origin

    @classmethod
    def field2updater(
        cls,
        pydantic_instance: BaseModel,
        field_name: str,
        **kwargs,
    ):
        """
        Builds a streamlit widget for a pydantic field.

        The output is saved to the pydantic_instance.
        """
        # Getting field value and info
        curr = getattr(pydantic_instance, field_name)
        field_info = pydantic_instance.model_fields[field_name]
        default = field_info.default
        # Getting type and nullable
        nullable, my_type = cls.check_type_n_nullable(field_info.annotation)
        args = get_args(field_info.annotation)

        # If a tuple, then building tuple inputs
        output = None
        if issubclass(my_type, tuple):
            # Building tuple inputs
            funcs_ls = []
            nullable_ls = []
            for i, arg in enumerate(args):
                nullable, my_type = cls.check_type_n_nullable(arg)
                funcs_ls.append(cls.type2updater(my_type))
                nullable_ls.append(nullable)
            output = cls.tuple_inputs(
                label=field_name,
                n=len(curr),
                func=tuple(funcs_ls),
                nullable=tuple(nullable_ls),
                default=default,
                curr=curr,
                **kwargs,
            )
        else:
            # Otherwise, using type2updated immediately
            output = cls.type2updater(my_type)(
                label=field_name,
                curr=curr,
                nullable=nullable,
                default=default,
                **kwargs,
            )
        # Setting to pydantic_instance
        setattr(pydantic_instance, field_name, output)

    @classmethod
    def field2updater_nice(
        cls,
        pydantic_instance: BaseModel,
        field_name: str,
        **kwargs,
    ):
        n_labels = SUBLABEL_MAP.get(field_name, None)
        cls.field2updater(
            pydantic_instance=pydantic_instance,
            field_name=field_name,
            n_labels=n_labels,
            **kwargs,
        )


@staticmethod
def configs_reset_func():
    """
    For each config parameter, resets the value to the value from disk.

    Also updates the session state variable that are dependent on this value:
    - `{label}`: the value
    - `{label}_is_none`: whether the value is None
    """
    # Loading and getting configs from disk
    load_configs()
    configs: ConfigParamsModel = st.session_state[CONFIGS]
    # For each field, setting the value to the value from disk
    for label, value in configs.model_dump().items():
        if isinstance(value, tuple):
            # If field is a tuple, then setting each value in tuple separately
            # Expects an entry in SUBLABEL_MAP
            for i, sublabel in enumerate(SUBLABEL_MAP[label]):
                label_value = f"{UPDATE}_{label}_{sublabel}_{VALUE}"
                label_is_none = f"{UPDATE}_{label}_{sublabel}_is_none"
                st.session_state[label_value] = value[i]
                st.session_state[label_is_none] = value[i] is None
        else:
            # Otherwise, setting the value directly
            label_value = f"{UPDATE}_{label}_{VALUE}"
            label_is_none = f"{UPDATE}_{label}_is_none"
            st.session_state[label_value] = value
            st.session_state[label_is_none] = value is None


def configs_save_func():
    """
    Saving configs from session state to project directory.

    NOTE: does not catch errors
    """
    configs: ConfigParamsModel = st.session_state[CONFIGS]
    proj_dir = st.session_state[PROJ_DIR]
    pfm = get_proj_fp_model(proj_dir)
    fp = pfm.config_params
    write_json(fp, configs.model_dump())


@page_decorator()
def page2_configs():
    """
    Displays and allows editing of configuration parameters for the project.
    This function uses Streamlit to create an interactive GUI for editing various
    configuration parameters stored in the session state. The parameters are grouped
    into different categories such as Reference, Raw, Registration, Mask, Overlap,
    and Cell Counting. Each category is expandable and allows the user to update
    specific fields.

    The function performs the following tasks:
    1. Retrieves the current configuration parameters from the session state.
    2. Displays the configuration parameters in expandable sections for editing.
    3. Validates and updates the configuration parameters in the session state.
    4. Provides a button to save the updated configuration parameters.
    The configuration parameters are instances of the `ConfigParamsModel` class.

    Note
    ----
        The function assumes that `ConfigsUpdater` and `ConfigParamsModel` are
        defined elsewhere in the codebase.

    Raises
    ------
        ValidationError: If the configuration parameters do not pass validation.
    """
    # Initialising session state variables

    # Recalling session state variables
    configs: ConfigParamsModel = st.session_state[CONFIGS]

    st.write("# Edit Configs")
    with st.expander("Reference"):
        ConfigsUpdater.field2updater_nice(configs, "atlas_dir")
        ConfigsUpdater.field2updater_nice(configs, "ref_v")
        ConfigsUpdater.field2updater_nice(configs, "annot_v")
        ConfigsUpdater.field2updater_nice(configs, "map_v")
    with st.expander("Raw"):
        ConfigsUpdater.field2updater_nice(configs, "chunksize")
    with st.expander("Registration"):
        # TODO: numerical is unintuitive for selecting axes in ref_orienf_ls
        ConfigsUpdater.field2updater_nice(configs, "ref_orient_ls")
        ConfigsUpdater.field2updater_nice(configs, "ref_z_trim")
        ConfigsUpdater.field2updater_nice(configs, "ref_y_trim")
        ConfigsUpdater.field2updater_nice(configs, "ref_x_trim")
        ConfigsUpdater.field2updater_nice(configs, "z_rough")
        ConfigsUpdater.field2updater_nice(configs, "y_rough")
        ConfigsUpdater.field2updater_nice(configs, "x_rough")
        ConfigsUpdater.field2updater_nice(configs, "z_fine")
        ConfigsUpdater.field2updater_nice(configs, "y_fine")
        ConfigsUpdater.field2updater_nice(configs, "x_fine")
        ConfigsUpdater.field2updater_nice(configs, "z_trim")
        ConfigsUpdater.field2updater_nice(configs, "y_trim")
        ConfigsUpdater.field2updater_nice(configs, "x_trim")
    with st.expander("Mask"):
        ConfigsUpdater.field2updater_nice(configs, "mask_gaus_blur")
        ConfigsUpdater.field2updater_nice(configs, "mask_thresh")
    with st.expander("Overlap"):
        ConfigsUpdater.field2updater_nice(configs, "depth")
    with st.expander("Cell Counting"):
        ConfigsUpdater.field2updater_nice(configs, "tophat_sigma")
        ConfigsUpdater.field2updater_nice(configs, "dog_sigma1")
        ConfigsUpdater.field2updater_nice(configs, "dog_sigma2")
        ConfigsUpdater.field2updater_nice(configs, "gauss_sigma")
        ConfigsUpdater.field2updater_nice(configs, "thresh_p")
        ConfigsUpdater.field2updater_nice(configs, "min_threshd")
        ConfigsUpdater.field2updater_nice(configs, "max_threshd")
        ConfigsUpdater.field2updater_nice(configs, "maxima_sigma")
        ConfigsUpdater.field2updater_nice(configs, "min_wshed")
        ConfigsUpdater.field2updater_nice(configs, "max_wshed")

    # Checking configs and updating in session_state
    # NOTE: an error can occur with the validation here
    configs = ConfigParamsModel.model_validate(configs)
    st.session_state[CONFIGS] = configs

    # Showing updated configs
    st.write("# See Project Configs")
    with st.expander("See Configs"):
        # JSON configs
        st.json(configs.model_dump())

    columns = st.columns(2)
    with columns[0]:
        # Button: Reset to old saved configs
        st.button(
            label="Reset",
            on_click=configs_reset_func,
            key=CONFIGS_RESET,
        )
        if st.session_state[CONFIGS_RESET]:
            st.success("Resetted project directory and configs")
    with columns[1]:
        # Button: Save new configs
        columns[1].button(
            label="Save",
            on_click=configs_save_func,
            key=CONFIGS_SAVE,
        )
        if st.session_state[CONFIGS_SAVE]:
            st.success("New configs saved to project directory")
