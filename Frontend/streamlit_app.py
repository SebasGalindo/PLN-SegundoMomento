import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Taller Segundo Momento AG"),
        st.Page("chatbot_streamlit.py", title="Chatbot"),
    ],
    "Explicaciones": [
        st.Page("explanation.py", title="Explicación del Chatbot"),
    ],
}

pg = st.navigation(pages)
pg.run()

