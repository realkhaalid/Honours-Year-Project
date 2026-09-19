from io import BytesIO
from pathlib import Path
import hashlib
import tempfile
import time

import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import streamlit as st
import torch

from data_processing_pipeline import (
    load_and_validate_audio,
    find_valid_subsamples,
    calculate_loudness_db,
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram,
    CLIP_DURATION_SECONDS,
    LOUDNESS_THRESHOLD_DB
)

from inference_pipeline import InferencePipeline


# Page Configuration
st.set_page_config(
    page_title="Audio Transformer",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# Configuration
MODEL_OPTIONS = {
    "MEL - Pretrained + Fine-Tuned": {
        "checkpoint":
            "checkpoints/"
            "mel_pretrained_fine_tuned_test_supervised_latest.pt",

        "representation":
            "MEL"
    },

    "MEL - Supervised Only": {
        "checkpoint":
            "checkpoints/"
            "mel_supervised_only_test_supervised_best.pt",

        "representation":
            "MEL"
    }
}


REPRESENTATION_FUNCTIONS = {
    "MEL":
        convert_to_mel_spectrogram,

    "STFT":
        convert_to_stft_spectrogram,

    "CQT":
        convert_to_cqt_spectrogram
}

# Utility Functions
def save_uploaded_file(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temporary_directory = Path(
        tempfile.gettempdir()
    ) / "audio_transformer"

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        temporary_directory
        / f"{file_hash}{suffix}"
    )

    if not file_path.exists():

        file_path.write_bytes(
            file_bytes
        )

    return file_path


def audio_to_bytes(
    audio,
    sample_rate
):

    buffer = BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV"
    )

    buffer.seek(0)

    return buffer.getvalue()


@st.cache_data(
    show_spinner=False
)
def process_uploaded_audio(
    file_path_string
):

    file_path = Path(
        file_path_string
    )

    audio, sample_rate = (
        load_and_validate_audio(
            file_path
        )
    )

    (
        valid_start_samples,
        statistics
    ) = (
        find_valid_subsamples(
            audio,
            sample_rate
        )
    )

    clip_length = int(
        sample_rate
        * CLIP_DURATION_SECONDS
    )

    all_samples = []

    valid_start_set = set(
        valid_start_samples
    )

    for start_sample in range(
        0,
        len(audio),
        clip_length
    ):

        end_sample = (
            start_sample
            + clip_length
        )

        sample = audio[
            start_sample:end_sample
        ]

        if len(sample) != clip_length:
            continue

        loudness = (
            calculate_loudness_db(
                sample
            )
        )

        all_samples.append({
            "start_sample":
                start_sample,

            "end_sample":
                end_sample,

            "audio":
                sample,

            "loudness":
                loudness,

            "valid":
                start_sample
                in valid_start_set
        })

    return (
        audio,
        sample_rate,
        all_samples,
        statistics
    )


@st.cache_resource(
    show_spinner=False
)
def get_inference_pipeline(
    checkpoint_path,
    representation_name
):

    representation_function = (
        REPRESENTATION_FUNCTIONS[
            representation_name
        ]
    )

    pipeline = InferencePipeline(
        checkpoint_path=(
            Path(
                checkpoint_path
            )
        ),
        data_representation=(
            representation_function
        )
    )

    return pipeline


def create_spectrogram_figure(
    audio,
    sample_rate,
    representation_name
):

    representation_function = (
        REPRESENTATION_FUNCTIONS[
            representation_name
        ]
    )

    spectrogram = (
        representation_function(
            audio,
            sample_rate
        )
        .detach()
        .cpu()
        .numpy()
    )

    figure, axis = plt.subplots(
        figsize=(12, 4)
    )

    if representation_name == "MEL":

        image = librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=axis
        )

        axis.set_title(
            "Log-MEL Spectrogram"
        )

    elif representation_name == "STFT":

        image = librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="linear",
            ax=axis
        )

        axis.set_title(
            "Log-STFT Spectrogram"
        )

    else:

        image = librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="cqt_hz",
            ax=axis
        )

        axis.set_title(
            "Log-CQT Spectrogram"
        )

    figure.colorbar(
        image,
        ax=axis,
        format="%+2.0f dB"
    )

    figure.tight_layout()

    return figure

# Upload Audio
st.title(
    "Audio Classification System"
)

st.write(
    "Upload an audio file to inspect the preprocessing "
    "pipeline and run Transformer inference."
)

uploaded_file = st.file_uploader(
    "Select Audio File",
    type=[
        "wav",
        "flac"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a WAV or FLAC audio file to begin."
    )

    st.stop()


audio_file_path = save_uploaded_file(
    uploaded_file
)

# Load Audio
try:

    (
        audio,
        sample_rate,
        all_samples,
        processing_statistics
    ) = process_uploaded_audio(
        str(
            audio_file_path
        )
    )

except ValueError as error:

    st.error(
        f"Audio processing failed: {error}"
    )

    st.stop()


duration = (
    len(audio)
    / sample_rate
)

# Audio Preview
st.subheader(
    "Audio Preview"
)

preview_col1, preview_col2 = st.columns(
    [
        2,
        1
    ]
)

with preview_col1:

    st.audio(
        uploaded_file.getvalue()
    )

with preview_col2:

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Duration:** {duration:.2f} seconds"
    )

    st.write(
        f"**Sample Rate:** {sample_rate} Hz"
    )

# Data Processing Demo
st.markdown(
    '<div id="data-processing-demo"></div>',
    unsafe_allow_html=True
)

st.header(
    "Data Processing Demo"
)

st.write(
    "This section demonstrates how the uploaded audio "
    "is divided into 5-second samples and filtered before "
    "being converted into the selected spectrogram "
    "representation."
)

# Pipeline Overview
pipeline_col1, arrow_col1, pipeline_col2, arrow_col2, pipeline_col3 = (
    st.columns(
        [
            3,
            1,
            3,
            1,
            3
        ]
    )
)

with pipeline_col1:

    st.markdown(
        f"""
        <div class="pipeline-box">
            <strong>Audio File</strong>
            <br><br>
            {duration:.2f} seconds
        </div>
        """,
        unsafe_allow_html=True
    )

with arrow_col1:

    st.markdown(
        '<div class="pipeline-arrow">→</div>',
        unsafe_allow_html=True
    )

with pipeline_col2:

    st.markdown(
        f"""
        <div class="pipeline-box">
            <strong>Split Audio</strong>
            <br><br>
            {processing_statistics["total_sub_samples"]} samples
        </div>
        """,
        unsafe_allow_html=True
    )

with arrow_col2:

    st.markdown(
        '<div class="pipeline-arrow">→</div>',
        unsafe_allow_html=True
    )

with pipeline_col3:

    st.markdown(
        f"""
        <div class="pipeline-box">
            <strong>Filtered Audio</strong>
            <br><br>
            {processing_statistics["valid_sub_samples"]} samples
        </div>
        """,
        unsafe_allow_html=True
    )

# Processing Statistics
st.subheader(
    "Processing Statistics"
)

stats_col1, stats_col2, stats_col3 = (
    st.columns(3)
)

with stats_col1:

    st.metric(
        "Total Samples",
        processing_statistics[
            "total_sub_samples"
        ]
    )

with stats_col2:

    st.metric(
        "Valid Samples",
        processing_statistics[
            "valid_sub_samples"
        ]
    )

with stats_col3:

    st.metric(
        "Filtered Samples",
        processing_statistics[
            "quiet_sub_samples"
        ]
    )

# Sample Explorer
st.subheader(
    "Audio Sample Explorer"
)

if len(all_samples) == 0:

    st.warning(
        "No complete 5-second audio samples were found."
    )

else:

    show_samples = st.radio(
        "Samples to display",
        [
            "Filtered / Valid Samples",
            "All Split Samples"
        ],
        horizontal=True
    )

    if (
        show_samples
        == "Filtered / Valid Samples"
    ):

        displayed_samples = [
            sample
            for sample
            in all_samples
            if sample[
                "valid"
            ]
        ]

    else:

        displayed_samples = (
            all_samples
        )

    if len(displayed_samples) == 0:

        st.warning(
            "No samples are available in this category."
        )

    else:

        sample_options = {}

        for index, sample in enumerate(
            displayed_samples,
            start=1
        ):

            start_time = (
                sample[
                    "start_sample"
                ]
                / sample_rate
            )

            end_time = (
                sample[
                    "end_sample"
                ]
                / sample_rate
            )

            status = (
                "Valid"
                if sample[
                    "valid"
                ]
                else "Filtered"
            )

            label = (
                f"Sample {index} | "
                f"{start_time:.1f}s - "
                f"{end_time:.1f}s | "
                f"{status}"
            )

            sample_options[
                label
            ] = sample

        selected_sample_label = (
            st.selectbox(
                "Select audio sample",
                list(
                    sample_options.keys()
                )
            )
        )

        selected_sample = (
            sample_options[
                selected_sample_label
            ]
        )

        sample_info_col1, sample_info_col2 = (
            st.columns(2)
        )

        with sample_info_col1:

            st.audio(
                audio_to_bytes(
                    selected_sample[
                        "audio"
                    ],
                    sample_rate
                ),
                format="audio/wav"
            )

        with sample_info_col2:

            st.metric(
                "Loudness",
                (
                    f"{selected_sample['loudness']:.2f} "
                    f"dBFS"
                )
            )

            st.write(
                "**Filter threshold:** "
                f"{LOUDNESS_THRESHOLD_DB:.2f} dBFS"
            )

            st.write(
                "**Status:**",
                (
                    "Valid"
                    if selected_sample[
                        "valid"
                    ]
                    else "Filtered"
                )
            )

        # Spectrogram Visualisation
        st.subheader(
            "Spectrogram Visualisation"
        )

        representation_name = (
            st.radio(
                "Representation",
                [
                    "MEL",
                    "STFT",
                    "CQT"
                ],
                horizontal=True
            )
        )

        with st.spinner(
            f"Generating {representation_name} spectrogram..."
        ):

            figure = (
                create_spectrogram_figure(
                    selected_sample[
                        "audio"
                    ],
                    sample_rate,
                    representation_name
                )
            )

            st.pyplot(
                figure,
                use_container_width=True
            )

            plt.close(
                figure
            )

# Inference
st.markdown(
    '<div id="inference"></div>',
    unsafe_allow_html=True
)

st.header(
    "Inference"
)

st.write(
    "Select a trained Transformer model and run "
    "classification on the uploaded audio."
)

# Input and Model
input_col, model_col = st.columns(
    2
)

with input_col:

    st.subheader(
        "Input Audio File"
    )

    st.write(
        f"**{uploaded_file.name}**"
    )

    st.audio(
        uploaded_file.getvalue()
    )

    st.caption(
        f"{duration:.2f} seconds | "
        f"{sample_rate} Hz"
    )


with model_col:

    st.subheader(
        "Select Model"
    )

    selected_model_name = (
        st.selectbox(
            "Available Models",
            list(
                MODEL_OPTIONS.keys()
            )
        )
    )

    selected_model = (
        MODEL_OPTIONS[
            selected_model_name
        ]
    )

    st.write(
        f"**Model:** {selected_model_name}"
    )

    st.write(
        "**Input Representation:** "
        f"{selected_model['representation']}"
    )

    st.write(
        "**Task:** Instrument Classification"
    )

    st.write(
        "**Architecture:** Transformer"
    )

# Run Inference
st.subheader(
    "Run Inference"
)

run_inference = st.button(
    "Run Inference",
    type="primary",
    use_container_width=True
)


if run_inference:

    checkpoint_path = (
        selected_model[
            "checkpoint"
        ]
    )

    representation_name = (
        selected_model[
            "representation"
        ]
    )

    if not Path(
        checkpoint_path
    ).exists():

        st.error(
            "Checkpoint could not be found: "
            f"{checkpoint_path}"
        )

    else:

        try:

            inference_pipeline = (
                get_inference_pipeline(
                    checkpoint_path,
                    representation_name
                )
            )

            start_time = (
                time.perf_counter()
            )

            with st.spinner(
                "Running inference..."
            ):

                results = (
                    inference_pipeline.predict(
                        audio_file_path
                    )
                )

            inference_time = (
                time.perf_counter()
                - start_time
            )

            st.session_state[
                "inference_results"
            ] = results

            st.session_state[
                "inference_time"
            ] = inference_time

            st.session_state[
                "inference_model"
            ] = selected_model_name

            st.session_state[
                "inference_representation"
            ] = representation_name

        except Exception as error:

            st.error(
                f"Inference failed: {error}"
            )

# Inference Results
if (
    "inference_results"
    in st.session_state
):

    results = (
        st.session_state[
            "inference_results"
        ]
    )

    st.subheader(
        "Inference Results"
    )

    prediction_col, confidence_col = (
        st.columns(2)
    )

    with prediction_col:

        st.metric(
            "Top Prediction",
            results[
                "predicted_label"
            ]
        )

    with confidence_col:

        st.metric(
            "Confidence",
            (
                f"{results['confidence'] * 100:.2f}%"
            )
        )

    # Class Probabilities
    st.subheader(
        "Class Probabilities"
    )

    probability_dataframe = pd.DataFrame(
        [
            {
                "Instrument":
                    label,

                "Probability":
                    probability * 100
            }

            for label, probability
            in results[
                "class_probabilities"
            ].items()
        ]
    )

    probability_dataframe = (
        probability_dataframe
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    st.bar_chart(
        probability_dataframe,
        x="Instrument",
        y="Probability",
        horizontal=True
    )

    st.dataframe(
        probability_dataframe.style.format({
            "Probability":
                "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Additional Information
    st.subheader(
        "Additional Information"
    )

    info_col1, info_col2, info_col3 = (
        st.columns(3)
    )

    with info_col1:

        st.write(
            "**Model Used**"
        )

        st.write(
            st.session_state[
                "inference_model"
            ]
        )

    with info_col2:

        st.write(
            "**Input Representation**"
        )

        st.write(
            st.session_state[
                "inference_representation"
            ]
        )

    with info_col3:

        st.write(
            "**Inference Time**"
        )

        st.write(
            f"{st.session_state['inference_time']:.3f} seconds"
        )

    # Per-Sample Predictions
    with st.expander(
        "5-Second Sample Predictions"
    ):

        sample_dataframe = (
            pd.DataFrame(
                results[
                    "sample_predictions"
                ]
            )
        )

        sample_dataframe[
            "confidence"
        ] = (
            sample_dataframe[
                "confidence"
            ]
            * 100
        )

        sample_dataframe = (
            sample_dataframe.rename(
                columns={
                    "sample":
                        "Sample",

                    "predicted_index":
                        "Class Index",

                    "predicted_label":
                        "Prediction",

                    "confidence":
                        "Confidence"
                }
            )
        )

        st.dataframe(
            sample_dataframe.style.format({
                "Confidence":
                    "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )