import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from text import instruction

# Inicialización de Vertex AI con las credenciales de tu proyecto
vertexai.init(project="miniibex-project", location="us-central1")

# Texto del sistema de instrucción
textsi_1 = instruction

# Configuraciones de generación y seguridad
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Función para generar contenido usando la API de Google Cloud


def multiturn_generate_content(text, container):
    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=[textsi_1]
    )
    chat = model.start_chat(response_validation=False)
    full_summary = ""
    for response in chat.send_message(text, generation_config=generation_config, safety_settings=safety_settings, stream=True):
        full_summary += response.candidates[0].content.parts[0].text
        container.markdown(full_summary, unsafe_allow_html=True)
    return full_summary


def main():
    st.title("Extractor de palabras relevantes e información importante")

    # Área de texto para que el usuario ingrese el texto
    user_input = st.text_area(
        "Ingresa el texto aquí", height=300)

    if st.button("Generar Respuesta"):
        if user_input:
            st.session_state.summary_container = st.empty()
            summary = multiturn_generate_content(
                user_input, st.session_state.summary_container)
        else:
            st.error("Por favor ingresa algún texto antes de generar la respuesta.")


if __name__ == "__main__":
    main()
