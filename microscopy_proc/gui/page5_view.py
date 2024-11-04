import os
from copy import deepcopy
from enum import Enum

import dask.array as da
import numpy as np
import streamlit as st

from microscopy_proc.constants import Coords
from microscopy_proc.funcs.viewer_funcs import CMAP as CMAP_D
from microscopy_proc.funcs.viewer_funcs import IMGS as IMGS_D
from microscopy_proc.funcs.viewer_funcs import VRANGE as VRANGE_D
from microscopy_proc.funcs.viewer_funcs import view_arrs_mp
from microscopy_proc.gui.gui_funcs import PROJ_DIR, SliceNames, init_var, page_decorator
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

# NOTE: could plt.colourmaps() work?
VIEW = "viewer"
GROUPS = f"{VIEW}_groups"
IMGS = f"{VIEW}_imgs"
TRIMMER = f"{VIEW}_trimmer"
NAME = f"{VIEW}_name"
VRANGE = f"{VIEW}_vrange"
CMAP = f"{VIEW}_cmap"
SEL = f"{VIEW}_sel"
RUN = f"{VIEW}_visualiser_run"


class Colormaps(Enum):
    GRAY = "gray"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    SET1 = "Set1"


def trimmer_func(coord):
    # Updating own input variable
    st.session_state[TRIMMER][coord] = st.session_state[f"{TRIMMER}_{coord}_w"]


@page_decorator()
def page5_view():
    # Initialising session state variables
    init_var(IMGS, deepcopy(IMGS_D))
    init_var(TRIMMER, {coord: slice(None) for coord in Coords})

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = get_proj_fp_model(proj_dir)

    st.write("## Visualiser")
    # Checking the max dimensions for trimmer sliders
    arr = None
    for i in [pfm.overlap, pfm.raw]:
        try:
            arr = da.from_zarr(pfm.overlap)
            break
        except Exception:
            st.warning(f"No {i} file found")
    # Making trimmer sliders if array exists
    if arr is not None:
        for i, coord in enumerate(Coords):
            # Making slider
            st.slider(
                label=f"{coord.value} trimmer",
                min_value=0,
                max_value=arr.shape[i],
                step=10,
                value=st.session_state[TRIMMER][coord],
                on_change=trimmer_func,
                args=(coord,),
                key=f"{TRIMMER}_{coord}_w",
            )

    else:
        # Otherwise arr is None
        # Warning
        st.error(
            "No overlap or raw array files found.\n\n"
            + "No trimming is available (if image too big this may crash the application)."
        )
        # trimmers are set to None
        st.write("No Z trimming")
        st.write("No Y trimming")
        st.write("No X trimming")
    # Convenience storing trimmer session state variable
    trimmer = st.session_state[TRIMMER]
    # Making visualiser checkboxes for each array
    visualiser_imgs = st.session_state[IMGS]
    for group_k, group_v in visualiser_imgs.items():
        with st.expander(f"{group_k}"):
            for img_k, img_v in group_v.items():
                # Initialising IMGS dict values for each image
                img_v[SEL] = img_v.get(SEL, False)
                img_v[VRANGE] = img_v.get(VRANGE, img_v[VRANGE_D])
                img_v[CMAP] = img_v.get(CMAP, img_v[CMAP_D])
                with st.container(border=True):
                    st.write(img_k)
                    # Checking if image file exists
                    if os.path.exists(getattr(pfm, img_k)):
                        columns = st.columns(3)
                        img_v[SEL] = columns[0].checkbox(
                            label="view image",
                            value=img_v[SEL],
                            key=f"{SEL}_{img_k}",
                        )
                        img_v[VRANGE] = columns[1].slider(
                            label="intensity range",
                            min_value=img_v[VRANGE_D][0],
                            max_value=img_v[VRANGE_D][1],
                            value=img_v[VRANGE],
                            disabled=not img_v[SEL],
                            key=f"{VRANGE}_{img_k}",
                        )
                        img_v[CMAP] = columns[2].selectbox(
                            label="colourmap",
                            options=enum2list(Colormaps),
                            index=enum2list(Colormaps).index(img_v[CMAP]),
                            disabled=not img_v[SEL],
                            key=f"{CMAP}_{img_k}",
                        )
                    else:
                        # If image file does not exist, then display warning
                        st.warning(f"Image file {img_k} does not exist")
    # Getting list of selected images (and each image's configs)
    imgs_to_run_ls = []
    for group_k, group_v in visualiser_imgs.items():
        for img_k, img_v in group_v.items():
            if img_v[SEL]:
                imgs_to_run_ls.append({NAME: img_k, GROUPS: group_k, **img_v})
    # Button: run visualiser
    st.button(
        label="Run visualiser",
        key=RUN,
    )
    if st.session_state[RUN]:
        # Showing selected visualiser
        st.write("### Running visualiser")
        st.write(
            "With trim of:\n"
            + f" - Z trim: {trimmer[0].start or SliceNames.START.value} - {trimmer[0].stop or SliceNames.STOP.value}\n"
            + f" - Y trim: {trimmer[1].start or SliceNames.START.value} - {trimmer[1].stop or SliceNames.STOP.value}\n"
            + f" - X trim: {trimmer[2].start or SliceNames.START.value} - {trimmer[2].stop or SliceNames.STOP.value}\n"
        )
        # Writing description of current image
        for img_v in imgs_to_run_ls:
            st.write(
                f"- Showing {img_v[GROUPS]} - {img_v[NAME]}\n"
                + f"    - intensity range: {img_v[VRANGE][0]} - {img_v[VRANGE][1]}\n"
                + f"    - colourmap: {img_v[CMAP]}\n"
            )
        # Running visualiser
        view_arrs_mp(
            fp_ls=tuple(getattr(pfm, i[NAME]) for i in imgs_to_run_ls),
            trimmer=trimmer,
            name=tuple(i[NAME] for i in imgs_to_run_ls),
            contrast_limits=tuple(i[VRANGE] for i in imgs_to_run_ls),
            colormap=tuple(i[CMAP] for i in imgs_to_run_ls),
        )

    # TODO: have an image saving function (from viewer_funcs) that can be called from here

    # Image size estimate
    # First checking if there are trimming dimensions
    if np.all([trimmer[coord].start is not None for coord in Coords]) and np.all(
        [trimmer[coord].start is not None for coord in Coords]
    ):
        # TODO: make more accurate with pixel data types (e.g. uint8, uint16, uint32)
        byte_size = (
            len(imgs_to_run_ls)
            * np.prod([trimmer[coord].stop - trimmer[coord].start for coord in Coords])
            * 16
        )
        byte_size_gb = byte_size / np.power(10, 9)
        st.write(f"Trim dimensions are {byte_size_gb} GB")
    else:
        # Otherwise outputting warning that trim dimensions are unknown
        st.warning(
            "Trim dimensions are unknown, "
            + "because overlap and raw arr do not exist in the project"
        )
