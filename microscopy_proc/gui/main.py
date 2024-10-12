import logging
import os
import subprocess

import streamlit as st

from microscopy_proc.gui.gui_funcs import ProjDirStatus
from microscopy_proc.gui.page_config import page_configs
from microscopy_proc.gui.page_init_proj import page_init_proj
from microscopy_proc.gui.page_pipeline import page_pipeline
from microscopy_proc.gui.page_visualiser import page_visualiser

logging.disable(logging.CRITICAL)


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
