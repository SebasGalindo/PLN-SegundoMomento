# chatbot_streamlit.py
import streamlit as st
import time

# Importa las funciones y estado inicial del módulo refactorizado
# Asegúrate de que chatbot_logic.py esté en el mismo directorio o en PYTHONPATH
try:
    import Backend.chatbot_logic as bot
except ModuleNotFoundError:
    st.error("Error: No se encontró el archivo 'chatbot_logic.py'. Asegúrate de que esté en el mismo directorio.")
    st.stop()
except Exception as e:
    # Mostrar un error más detallado durante el desarrollo puede ser útil
    st.error(f"Error al cargar 'chatbot_logic.py': {e}")
    import traceback
    st.error(traceback.format_exc()) # Muestra el traceback completo
    st.stop()


# --- Configuración de la Página Streamlit ---
st.set_page_config(page_title="Chatbot Financiero", layout="centered")
st.title("🤖 Chatbot Financiero")
st.caption("Impulsado por Modelos de IA para análisis preliminar")

# --- Barra Lateral con Botón de Reinicio ---
st.sidebar.title("Opciones")
if st.sidebar.button("✨ Nuevo Chat"):
    # Reiniciar el estado de la sesión y el historial
    st.session_state.chat_history = []
    st.session_state.chatbot_state = bot.get_initial_state()
    # Obtener el primer mensaje del bot para el nuevo chat
    initial_bot_message, updated_state = bot.get_initial_message(st.session_state.chatbot_state)
    st.session_state.chatbot_state = updated_state # Actualizar estado con last_question_field, etc.
    # Añadir el primer mensaje del bot al historial vacío
    st.session_state.chat_history.append({"role": "assistant", "content": initial_bot_message})
    # Forzar la re-ejecución del script para reflejar el reinicio
    st.rerun()

# --- Inicialización del Estado de la Sesión (Solo la primera vez) ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chatbot_state' not in st.session_state:
    st.session_state.chatbot_state = bot.get_initial_state()
    initial_bot_message, updated_state = bot.get_initial_message(st.session_state.chatbot_state)
    st.session_state.chatbot_state = updated_state
    st.session_state.chat_history.append({"role": "assistant", "content": initial_bot_message})

# --- Mostrar Historial del Chat ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Deshabilitar input si la conversación ha terminado ---
conversation_finished = st.session_state.chatbot_state.get('conversation_complete', False)

# --- Obtener Entrada del Usuario ---
user_input = st.chat_input(
    "Escribe tu mensaje aquí...",
    disabled=conversation_finished, # Deshabilita el input si la conversación terminó
    key="chat_input" # Añadir una clave puede ayudar a la gestión del estado
)

if conversation_finished and not user_input:
    st.info("Conversación finalizada. Presiona 'Nuevo Chat' en la barra lateral para comenzar de nuevo.")


if user_input:
    # Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Procesar la entrada del usuario usando la lógica del bot
    try:
        # Mostrar un indicador de "pensando"
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                bot_response, updated_state = bot.process_user_input(user_input, st.session_state.chatbot_state)
                st.session_state.chatbot_state = updated_state # Actualizar el estado global

            # Mostrar la respuesta del bot
            st.markdown(bot_response)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        # Verificar si la conversación se completó DESPUÉS de la respuesta
        # Y si la respuesta actual NO es ya el análisis final
        is_final_analysis_done = "Resultado del Análisis" in bot_response
        if st.session_state.chatbot_state.get('conversation_complete', False) and not is_final_analysis_done:
              with st.chat_message("assistant"):
                   with st.spinner("Realizando análisis final..."):
                        final_analysis_message = bot.get_final_analysis(st.session_state.chatbot_state)
                        time.sleep(1) # Pequeña pausa estética
                   st.markdown(final_analysis_message)
                   # Añadir el análisis final al historial también
                   st.session_state.chat_history.append({"role": "assistant", "content": final_analysis_message})
              # Forzar rerun para deshabilitar el input inmediatamente después del análisis
              st.rerun()

    except Exception as e:
        error_message = f"¡Ups! Ocurrió un error interno en el bot: {e}"
        st.error(error_message)
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        # Opcional: Mostrar traceback para depuración
        # import traceback
        # st.error(traceback.format_exc())

# --- Fin del archivo chatbot_streamlit.py ---