import streamlit as st

from .gui_funcs import page_decorator


@page_decorator()
def page_visualiser():
    st.write("## Visualiser")
