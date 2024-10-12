import os
import subprocess

import streamlit as st

from microscopy_proc.gui.gui_funcs import (
    ProjDirStatus,
)
from microscopy_proc.gui.page_config import page_configs
from microscopy_proc.gui.page_init_proj import page_init_proj
from microscopy_proc.gui.page_pipeline import page_pipeline
from microscopy_proc.gui.page_visualiser import page_visualiser

# from microscopy_proc.pipelines.pipeline_funcs import (
#     cell_mapping_pipeline,
#     cellc1_pipeline,
#     cellc2_pipeline,
#     cellc3_pipeline,
#     cellc4_pipeline,
#     cellc5_pipeline,
#     cellc6_pipeline,
#     cellc7_pipeline,
#     cellc8_pipeline,
#     cellc9_pipeline,
#     cellc10_pipeline,
#     cellc11_pipeline,
#     cellc_coords_only_pipeline,
#     cells2csv_pipeline,
#     group_cells_pipeline,
#     img_fine_pipeline,
#     img_overlap_pipeline,
#     img_rough_pipeline,
#     img_trim_pipeline,
#     make_mask_pipeline,
#     ref_prepare_pipeline,
#     registration_pipeline,
#     tiff2zarr_pipeline,
#     transform_coords_pipeline,
# )
# from microscopy_proc.scripts.gui_funcs import ConfigsUpdater, enum2list


# #####################################################################
# # Streamlit pages
# #####################################################################


#####################################################################
# Streamlit application
#####################################################################


def main():
    # Initialising session state
    if "proj_dir" not in st.session_state:
        st.session_state["proj_dir"] = None
        st.session_state["proj_dir_status"] = ProjDirStatus.NOT_SET
    # Title
    st.title("Microscopy Processing Pipeline")
    # Multi-page navigation
    pg = st.navigation(
        [
            st.Page(page_init_proj),
            st.Page(page_configs),
            st.Page(page_pipeline),
            st.Page(page_visualiser),
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
