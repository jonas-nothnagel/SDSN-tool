import logging
try:
    import streamlit as st
except ImportError:
    logging.info("Streamlit not installed")


def check_streamlit():
    """
    Function to check whether python code is run within streamlit

    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    try:
        from streamlit.scriptrunner.script_run_context import get_script_run_ctx
        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit

def disable_other_checkboxes(*other_checkboxes_keys):
    for checkbox_key in other_checkboxes_keys:
        st.session_state[checkbox_key] = False

def checkbox_without_preselect(keylist):
    dict_ = {}
    for i,key_val in enumerate(keylist):
        dict_[i] = st.checkbox(key_val,key = key_val,
        on_change = disable_other_checkboxes,
        args=tuple(list(filter(lambda x: x!= key_val, keylist))),)
    
    for key,val in dict_.items():
        if val == True:
            return keylist[int(key)]
    
    return None