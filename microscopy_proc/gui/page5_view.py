import os
from copy import deepcopy
from enum import Enum

import dask.array as da
import streamlit as st

from microscopy_proc.funcs.viewer_funcs import CMAP as CMAP_D
from microscopy_proc.funcs.viewer_funcs import IMGS as IMGS_D
from microscopy_proc.funcs.viewer_funcs import VRANGE as VRANGE_D
from microscopy_proc.funcs.viewer_funcs import view_arrs_mp
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import L_SLC, L_ZYX, PROJ_DIR, init_var, page_decorator

# NOTE: could plt.colourmaps() work?
VIEW = "viewer"
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


def trimmer_func(v):
    # Updating own input variable
    st.session_state[f"{TRIMMER}_{v}"] = st.session_state[f"{TRIMMER}_{v}_w"]


@page_decorator()
def page5_view():
    # Initialising session state variables
    init_var(IMGS, deepcopy(IMGS_D))

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = get_proj_fp_model(proj_dir)

    st.write("## Visualiser")
    # Checking the max dimensions for trimmer sliders
    arr = None
    try:
        arr = da.from_zarr(pfm.overlap)
    except Exception:
        try:
            arr = da.from_zarr(pfm.raw)
            st.warning("No overlap file found, using raw file instead")
        except Exception:
            st.error(
                "No overlap or raw array files found.\n\n"
                + "No trimming is available (if image too big this may crash application)."
            )
    # Initialising trimmer tuple
    trimmer = tuple(slice(None) for _ in L_ZYX)
    if arr is not None:
        # Making trimmer sliders if array exists
        for i, v in enumerate(L_ZYX):
            # Initialising trimmer value
            init_var(f"{TRIMMER}_{v}", (0, arr.shape[i]))
            # Making slider
            st.slider(
                label=f"{v} trimmer",
                min_value=0,
                max_value=arr.shape[i],
                step=10,
                value=st.session_state[f"{TRIMMER}_{v}"],
                on_change=trimmer_func,
                args=(v,),
                key=f"{TRIMMER}_{v}_w",
            )
        trimmer = tuple(slice(*st.session_state[f"{TRIMMER}_{v}"]) for v in L_ZYX)
    else:
        # Otherwise trimmers are set to None
        st.write("No Z trimming")
        st.write("No Y trimming")
        st.write("No X trimming")
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
            + f" - Z trim: {trimmer[0].start or L_SLC[0]} - {trimmer[0].stop or L_SLC[1]}\n"
            + f" - Y trim: {trimmer[1].start or L_SLC[0]} - {trimmer[1].stop or L_SLC[1]}\n"
            + f" - X trim: {trimmer[2].start or L_SLC[0]} - {trimmer[2].stop or L_SLC[1]}\n"
        )
        imgs_to_run_ls = []
        for group_k, group_v in visualiser_imgs.items():
            for img_k, img_v in group_v.items():
                if img_v[SEL]:
                    # Writing description of current image
                    st.write(
                        f"- Showing {group_k} - {img_k}\n"
                        + f"    - intensity range: {img_v[VRANGE][0]} - {img_v[VRANGE][1]}\n"
                        + f"    - colourmap: {img_v[CMAP]}\n"
                    )
                    # Also saving to imgs_to_run_ls list
                    imgs_to_run_ls.append({NAME: img_k, **img_v})
        # Running visualiser
        view_arrs_mp(
            fp_ls=tuple(getattr(pfm, i[NAME]) for i in imgs_to_run_ls),
            trimmer=trimmer,
            name=tuple(i[NAME] for i in imgs_to_run_ls),
            contrast_limits=tuple(i[VRANGE] for i in imgs_to_run_ls),
            colormap=tuple(i[CMAP] for i in imgs_to_run_ls),
        )


# TODO: have an image saving function (from viewer_funcs) that can be called from here
