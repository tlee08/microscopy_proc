import os

import dask.array as da
import streamlit as st

from microscopy_proc.funcs.viewer_funcs import view_arrs_mp
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

from .gui_funcs import PROJ_DIR, page_decorator

# NOTE: could plt.colourmaps() work?
CMAP = [
    "gray",
    "red",
    "green",
    "Set1",
]


@page_decorator()
def page_visualiser():
    if "visualiser_imgs" not in st.session_state:
        st.session_state["visualiser_imgs"] = {
            "Atlas": {
                "ref": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "green",
                    "sel": False,
                },
                "annot": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "Set1",
                    "sel": False,
                },
            },
            "Raw": {
                "raw": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "gray",
                    "sel": False,
                },
            },
            "Registration": {
                "downsmpl1": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "downsmpl2": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "trimmed": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "regresult": {
                    "vrange_default": (0, 1000),
                    "cmap_default": "green",
                    "sel": False,
                },
            },
            "Mask": {
                "premask_blur": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "red",
                    "sel": False,
                },
                "mask": {
                    "vrange_default": (0, 5),
                    "cmap_default": "red",
                    "sel": False,
                },
                "outline": {
                    "vrange_default": (0, 5),
                    "cmap_default": "red",
                    "sel": False,
                },
                "mask_reg": {
                    "vrange_default": (0, 5),
                    "cmap_default": "red",
                    "sel": False,
                },
            },
            "Cell Counting (overlapped)": {
                "overlap": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "bgrm": {
                    "vrange_default": (0, 2000),
                    "cmap_default": "green",
                    "sel": False,
                },
                "dog": {
                    "vrange_default": (0, 100),
                    "cmap_default": "red",
                    "sel": False,
                },
                "adaptv": {
                    "vrange_default": (0, 100),
                    "cmap_default": "red",
                    "sel": False,
                },
                "threshd": {
                    "vrange_default": (0, 5),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "threshd_volumes": {
                    "vrange_default": (0, 10000),
                    "cmap_default": "green",
                    "sel": False,
                },
                "threshd_filt": {
                    "vrange_default": (0, 5),
                    "cmap_default": "green",
                    "sel": False,
                },
                "maxima": {
                    "vrange_default": (0, 5),
                    "cmap_default": "green",
                    "sel": False,
                },
                "wshed_volumes": {
                    "vrange_default": (0, 1000),
                    "cmap_default": "green",
                    "sel": False,
                },
                "wshed_filt": {
                    "vrange_default": (0, 1000),
                    "cmap_default": "green",
                    "sel": False,
                },
            },
            "Cell Counting (trimmed)": {
                "threshd_final": {
                    "vrange_default": (0, 5),
                    "cmap_default": "gray",
                    "sel": False,
                },
                "maxima_final": {
                    "vrange_default": (0, 5),
                    "cmap_default": "red",
                    "sel": False,
                },
                "wshed_final": {
                    "vrange_default": (0, 1000),
                    "cmap_default": "green",
                    "sel": False,
                },
            },
            "Post Processing Checks": {
                "points_check": {
                    "vrange_default": (0, 5),
                    "cmap_default": "green",
                    "sel": False,
                },
                "heatmap_check": {
                    "vrange_default": (0, 20),
                    "cmap_default": "red",
                    "sel": False,
                },
                "points_trfm_check": {
                    "vrange_default": (0, 5),
                    "cmap_default": "green",
                    "sel": False,
                },
                "heatmap_trfm_check": {
                    "vrange_default": (0, 100),
                    "cmap_default": "red",
                    "sel": False,
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
                key=f"viewer_{v}_slicer",
            )
        sliders = [st.session_state[f"viewer_{v}_slicer"] for v in "ZYX"]
        trimmer = tuple(slice(*i) for i in sliders)
    else:
        # Otherwise slicers are set to None
        st.write("No Z trimming")
        st.write("No Y trimming")
        st.write("No X trimming")
    # Making visualiser checkboxes
    visualiser_imgs = st.session_state["visualiser_imgs"]
    for group_k, group_v in visualiser_imgs.items():
        with st.expander(f"{group_k}"):
            for img_k, img_v in group_v.items():
                with st.container(border=True):
                    st.write(img_k)
                    # Checking if image file exists
                    if os.path.exists(getattr(pfm, img_k)):
                        columns = st.columns(3)
                        img_v["sel"] = columns[0].checkbox(
                            label="view image",
                            value=img_v["sel"],
                            key=f"viewer_{img_k}_sel",
                        )
                        img_v["vrange"] = columns[1].slider(
                            label="intensity range",
                            min_value=img_v["vrange_default"][0],
                            max_value=img_v["vrange_default"][1],
                            value=img_v.get("vrange", img_v["vrange_default"]),
                            disabled=not img_v["sel"],
                            key=f"viewer_{img_k}_vrange",
                        )
                        img_v["cmap"] = columns[2].selectbox(
                            label="colourmap",
                            options=CMAP,
                            index=CMAP.index(img_v.get("cmap", img_v["cmap_default"])),
                            disabled=not img_v["sel"],
                            key=f"viewer_{img_k}_cmap",
                        )
                    else:
                        # If image file does not exist, then display warning
                        st.warning(f"Image file {img_k} does not exist")
    # Button: run visualiser
    st.button(
        label="Run visualiser",
        key="visualiser_run_btn",
    )
    if st.session_state["visualiser_run_btn"]:
        # Showing selected visualiser
        st.write("### Running visualiser")
        st.write(
            "With trim of:\n"
            + f" - Z trim: {trimmer[0].start or "start"} - {trimmer[0].stop or "end"}\n"
            + f" - Y trim: {trimmer[1].start or "start"} - {trimmer[1].stop or "end"}\n"
            + f" - X trim: {trimmer[2].start or "start"} - {trimmer[2].stop or "end"}\n"
        )
        imgs_to_run_ls = []
        for group_k, group_v in visualiser_imgs.items():
            for img_k, img_v in group_v.items():
                if img_v["sel"]:
                    st.write(
                        f"- Showing {group_k} - {img_k}\n"
                        + f"    - intensity range: {img_v["vrange"][0]} - {img_v["vrange"][1]}\n"
                        + f"    - colourmap: {img_v["cmap"]}\n"
                    )
                    # Also saving to imgs_to_run_ls list
                    imgs_to_run_ls.append({"name": img_k, **img_v})
        # Running visualiser
        view_arrs_mp(
            fp_ls=tuple(getattr(pfm, i["name"]) for i in imgs_to_run_ls),
            trimmer=trimmer,
            name=tuple(i["name"] for i in imgs_to_run_ls),
            contrast_limits=tuple(i["vrange"] for i in imgs_to_run_ls),
            colormap=tuple(i["cmap"] for i in imgs_to_run_ls),
        )
