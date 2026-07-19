import streamlit as st
import torch
from inference_testing import load_model, process_audio_file

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f6fc;
        }

        .main .block-container {
            max-width: 900px;
            padding-top: 2rem;
        }

        .screen-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
        }

        .header-icon {
            background-color: #eee9ff;
            color: #6138e8;
            width: 48px;
            height: 48px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .screen-title {
            font-size: 25px;
            font-weight: 700;
            margin: 0;
            color: #1d2433;
        }

        .screen-description {
            margin: 3px 0 0 0;
            color: #6c7280;
            font-size: 14px;
        }

        .upload-card {
            background-color: white;
            border: 1px solid #e6e3ef;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(40, 30, 80, 0.04);
        }

        .selected-title {
            color: #4b5261;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .file-card {
            background-color: white;
            border: 1px solid #e5e2ec;
            border-radius: 9px;
            padding: 14px;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .file-icon {
            background-color: #f2efff;
            color: #6339e8;
            width: 42px;
            height: 42px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
        }

        .file-name {
            font-weight: 600;
            color: #202633;
            margin-bottom: 3px;
        }

        .file-details {
            color: #737887;
            font-size: 13px;
        }

        .formats {
            color: #737887;
            font-size: 13px;
            margin-top: 16px;
        }

        div[data-testid="stFileUploader"] {
            border: 1.5px dashed #9a84f7;
            border-radius: 10px;
            padding: 25px;
            background-color: #fcfbff;
        }

        div[data-testid="stFileUploader"] section {
            padding: 20px 0;
        }

        div[data-testid="stFileUploader"] button {
            background-color: #6339e8;
            color: white;
            border: none;
            border-radius: 7px;
        }

        div[data-testid="stFileUploader"] button:hover {
            background-color: #5128d3;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "uploaded_audio" not in st.session_state:
    st.session_state.uploaded_audio = None

st.markdown(
    """
    <div class="screen-header">
        <div class="header-icon">🎵</div>
        <div>
            <p class="screen-title">Select Audio File</p>
            <p class="screen-description">
                Choose an audio file from your local system to run inference.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Drag and drop an audio file here",
        type=["wav"],
        key="audio_file_uploader"
    )

    if uploaded_file is not None:
        if uploaded_file != st.session_state.uploaded_audio:
            st.session_state.inference_result = None

        st.session_state.uploaded_audio = uploaded_file

    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None

    if st.session_state.uploaded_audio is not None:
        audio_file = st.session_state.uploaded_audio

        file_size_mb = audio_file.size / (1024 * 1024)
        file_extension = audio_file.name.split(".")[-1].upper()

        st.markdown(
            '<p class="selected-title">Selected File</p>',
            unsafe_allow_html=True
        )

        file_info_column, remove_column = st.columns([8, 1])

        with file_info_column:
            st.markdown(
                f"""
                <div class="file-card">
                    <div class="file-icon">🎵</div>
                    <div>
                        <div class="file-name">{audio_file.name}</div>
                        <div class="file-details">
                            {file_size_mb:.2f} MB • {file_extension}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with remove_column:
            if st.button(
                "✕",
                help="Remove selected file",
                use_container_width=True
            ):
                st.session_state.uploaded_audio = None
                st.rerun()

        st.subheader("Audio Preview")
        st.audio(uploaded_file)

        if st.button(
        "Run Inference",
        type="primary",
        use_container_width=True
        ):
            with st.spinner("Running inference..."):
                model, class_names = load_model()
                patches = process_audio_file(audio_file)

                with torch.no_grad():
                    predicted_index, probabilities = model.classify(patches)

                predicted_index = predicted_index.item()
                predicted_class = class_names[predicted_index]
                confidence = probabilities[0, predicted_index].item()

                st.session_state.inference_result = {
                    "prediction": predicted_class,
                    "confidence": confidence
                }

        if st.session_state.inference_result is not None:
            result = st.session_state.inference_result

            st.subheader("Inference Results")

            result_column, confidence_column = st.columns(2)

            with result_column:
                st.metric(
                    label="Predicted Class",
                    value=result["prediction"]
                )

            with confidence_column:
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence'] * 100:.2f}%"
                )

    st.markdown(
        """
        <div class="formats">
            ⓘ Supported formats: .wav
        </div>
        """,
        unsafe_allow_html=True
    )