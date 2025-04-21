import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Taller Segundo Momento AG"),
        st.Page("chatbot_streamlit.py", title="Chatbot"),
    ],
    "Explicaciones": [
        st.Page("explanation.py", title="Explicación del Chatbot"),
        st.Page("explanation_train_lightgbm.py", title="Entrenamiento para LightGBM"),
        st.Page("explanation_dataset_lightgbm.py", title="Dataset para LightGBM"),
        
    ],
}

pg = st.navigation(pages)
pg.run()

