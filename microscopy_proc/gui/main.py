import logging
import os
import subprocess

import streamlit as st

from microscopy_proc.gui.gui_funcs import PROJ_DIR, PROJ_DIR_STATUS, ProjDirStatus
from microscopy_proc.gui.page1_init import page1_init
from microscopy_proc.gui.page2_config import page2_configs
from microscopy_proc.gui.page3_pipeline import page3_pipeline
from microscopy_proc.gui.page4_visualiser import page4_visualiser
from microscopy_proc.gui.page5_combine import page5_combine

logging.disable(logging.CRITICAL)


#####################################################################
# Streamlit application
#####################################################################


def main():
    # Initialising session state
    if PROJ_DIR not in st.session_state:
        st.session_state[PROJ_DIR] = None
        st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_SET
    # Title
    st.title("Microscopy Processing Pipeline")
    # Multi-page navigation
    pg = st.navigation(
        [
            st.Page(page1_init),
            st.Page(page2_configs),
            st.Page(page3_pipeline),
            st.Page(page4_visualiser),
            st.Page(page5_combine),
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
