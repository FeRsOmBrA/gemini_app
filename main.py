import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import json
import tempfile
import os

# Crear archivo JSON temporal con las credenciales de Streamlit


def create_temp_credentials_file():
    # Accede a las variables de secretos de Streamlit
    secrets = st.secrets
    credentials = {
        "type": secrets["type"],
        "project_id": secrets["project_id"],
        "private_key_id": secrets["private_key_id"],
        # Asegúrate de manejar correctamente los saltos de línea
        "private_key": secrets["private_key"].replace('\\n', '\n'),
        "client_email": secrets["client_email"],
        "client_id": secrets["client_id"],
        "auth_uri": secrets["auth_uri"],
        "token_uri": secrets["token_uri"],
        "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
        "client_x509_cert_url": secrets["client_x509_cert_url"],
        # Valor predeterminado si no existe en secretos
        "universe_domain": secrets.get("universe_domain", "googleapis.com")
    }

    # Crear archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_file_path = temp_file.name

    with open(temp_file_path, 'w') as f:
        json.dump(credentials, f, indent=4)

    temp_file.close()  # Asegúrate de cerrar el archivo

    return temp_file_path

# Inicializar Vertex AI con el archivo de credenciales temporal


def init_vertex_ai():
    credentials_path = create_temp_credentials_file()
    vertexai.init(project="miniibex-project",
                  location="us-central1", credentials=credentials_path)

    # Elimina el archivo temporal después de usarlo
    os.remove(credentials_path)


# Texto del sistema de instrucción
textsi_1 = "El texto del sistema de instrucción va aquí."

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

    # Inicializar Vertex AI
    init_vertex_ai()

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
