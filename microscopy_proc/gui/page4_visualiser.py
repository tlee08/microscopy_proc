import os
from enum import Enum

import dask.array as da
import streamlit as st

from microscopy_proc.funcs.viewer_funcs import view_arrs_mp
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import L_SLC, L_ZYX, PROJ_DIR, page_decorator

# NOTE: could plt.colourmaps() work?
VIEWER = "viewer"
IMGS = "visualiser_imgs"
TRIMMER = "trimmer"
NAME = "name"
VRANGE = "vrange"
VRANGE_D = "vrange_default"
CMAP = "cmap"
CMAP_D = "cmap_default"
SEL = "sel"
RUN = "visualiser_run"


class Colormaps(Enum):
    GRAY = "gray"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    SET1 = "Set1"


@page_decorator()
def page4_visualiser():
    if IMGS not in st.session_state:
        st.session_state[IMGS] = {
            "Atlas": {
                "ref": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "annot": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.SET1.value,
                    SEL: False,
                },
            },
            "Raw": {
                "raw": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
            },
            "Registration": {
                "downsmpl1": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "downsmpl2": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "trimmed": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "regresult": {
                    VRANGE_D: (0, 1000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
            },
            "Mask": {
                "premask_blur": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "mask": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "outline": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "mask_reg": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
            },
            "Cell Counting (overlapped)": {
                "overlap": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "bgrm": {
                    VRANGE_D: (0, 2000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "dog": {
                    VRANGE_D: (0, 100),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "adaptv": {
                    VRANGE_D: (0, 100),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "threshd": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "threshd_volumes": {
                    VRANGE_D: (0, 10000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "threshd_filt": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "maxima": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "wshed_volumes": {
                    VRANGE_D: (0, 1000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "wshed_filt": {
                    VRANGE_D: (0, 1000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
            },
            "Cell Counting (trimmed)": {
                "threshd_final": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GRAY.value,
                    SEL: False,
                },
                "maxima_final": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "wshed_final": {
                    VRANGE_D: (0, 1000),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
            },
            "Post Processing Checks": {
                "points_check": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "heatmap_check": {
                    VRANGE_D: (0, 20),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
                "points_trfm_check": {
                    VRANGE_D: (0, 5),
                    CMAP_D: Colormaps.GREEN.value,
                    SEL: False,
                },
                "heatmap_trfm_check": {
                    VRANGE_D: (0, 100),
                    CMAP_D: Colormaps.RED.value,
                    SEL: False,
                },
            },
        }

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
    # Making slicer/trimmer sliders
    trimmer = tuple(slice(None) for _ in "ZYX")
    if arr is not None:
        # Making slicer sliders if array exists
        for i, v in enumerate("ZYX"):
            st.slider(
                label=f"{v} trimmer",
                min_value=0,
                max_value=arr.shape[i],
                step=10,
                value=(0, arr.shape[i]),
                key=f"{VIEWER}_{v}_{TRIMMER}",
            )
        sliders = [st.session_state[f"{VIEWER}_{v}_{TRIMMER}"] for v in L_ZYX]
        trimmer = tuple(slice(*i) for i in sliders)
    else:
        # Otherwise slicers are set to None
        st.write("No Z trimming")
        st.write("No Y trimming")
        st.write("No X trimming")
    # Making visualiser checkboxes for each array
    visualiser_imgs = st.session_state[IMGS]
    for group_k, group_v in visualiser_imgs.items():
        with st.expander(f"{group_k}"):
            for img_k, img_v in group_v.items():
                with st.container(border=True):
                    st.write(img_k)
                    # Checking if image file exists
                    if os.path.exists(getattr(pfm, img_k)):
                        columns = st.columns(3)
                        img_v[SEL] = columns[0].checkbox(
                            label="view image",
                            value=img_v[SEL],
                            key=f"{VIEWER}_{img_k}_{SEL}",
                        )
                        img_v[VRANGE] = columns[1].slider(
                            label="intensity range",
                            min_value=img_v[VRANGE_D][0],
                            max_value=img_v[VRANGE_D][1],
                            value=img_v.get(VRANGE, img_v[VRANGE_D]),
                            disabled=not img_v[SEL],
                            key=f"{VIEWER}_{img_k}_{VRANGE}",
                        )
                        img_v[CMAP] = columns[2].selectbox(
                            label="colourmap",
                            options=CMAP,
                            index=CMAP.index(img_v.get(CMAP, img_v[CMAP_D])),
                            disabled=not img_v[SEL],
                            key=f"{VIEWER}{img_k}_{CMAP}",
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
