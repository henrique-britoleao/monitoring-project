import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys

@contextmanager
def st_redirect(src, dst):
    '''
    redirects the standard stream (can be stdout or stderr) to streamlit

    src: file object
    dst: streamlit display funtion
    '''
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stderr(dst):
    '''This will show the logging'''
    with st_redirect(sys.stderr, dst):
        yield
