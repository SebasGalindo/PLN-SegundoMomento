import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Taller Segundo Momento AG"),
        st.Page("chatbot_streamlit.py", title="Chatbot"),
    ],
    "Explicaciones": [
        st.Page("explanation_train_lightgbm.py", title="Entrenamiento para LightGBM"),
        st.Page("explanation_dataset_lightgbm.py", title="Dataset para LightGBM"),
        st.Page("explanation_dataset_phrases.py", title="Dataset para BERT (Frases)"),
        st.Page("explanation_beto_phrases.py", title="BETO (Frases)"),
        st.Page("explanation_dataset_NER.py", title="Dataset para BERT (NER)"),
        st.Page("explanation_beto_NER.py", title="BETO (NER)"),
        st.Page("explanation_chatbot.py", title="Explicación del Chatbot"),
    ],
}

pg = st.navigation(pages)
pg.run()

