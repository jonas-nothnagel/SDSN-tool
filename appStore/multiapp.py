"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from utils.uploadAndExample import add_upload

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = []

    def add_app(self,title,icon, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "icon": icon,
            "function": func
        })

    def run(self):
        
        st.sidebar.write(format_func=lambda app: app['title'])
        image = Image.open('docStore/img/sdsn.png')
        st.sidebar.image(image, width =200)
       
        with st.sidebar:
            selected = option_menu(None, [page["title"] for page in self.apps],
                                   icons=[page["icon"] for page in self.apps],
                                   menu_icon="cast", default_index=0)
            st.markdown("---")          
        
        
        for index, item in enumerate(self.apps):
            if item["title"] == selected:
                self.apps[index]["function"]()
                break
                
   
        choice = st.sidebar.radio(label = 'Select the Document',
                            help = 'You can upload the document \
                            or else you can try a example document', 
                            options = ('Upload Document', 'Try Example'), 
                            horizontal = True)
        add_upload(choice)
       