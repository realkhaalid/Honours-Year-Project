from io import BytesIO
from pathlib import Path
import time

import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import streamlit as st

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

from multi_label_inference import (
    MultiLabelInferencePipeline
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Label Audio Transformer",
    page_icon="🎵",
    layout="wide"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_OPTIONS = {
    "MEL - Pretrained + Fine-Tuned": {
        "checkpoint":
            "multi_label_checkpoints/"
            "mel_multi_label_pretrained_fine_tuned_supervised_best.pt",

        "representation":
            "MEL"
    }
}


MULTI_LABEL_EXAMPLES_DIRECTORY = Path(
    "multi_label_examples"
)


REPRESENTATION_FUNCTIONS = {
    "MEL":
        convert_to_mel_spectrogram,

    "STFT":
        convert_to_stft_spectrogram,

    "CQT":
        convert_to_cqt_spectrogram
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

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

    buffer.seek(
        0
    )

    return buffer.getvalue()


def get_example_track_folders():

    if not MULTI_LABEL_EXAMPLES_DIRECTORY.exists():

        raise FileNotFoundError(
            "multi_label_examples folder could not be found: "
            f"{MULTI_LABEL_EXAMPLES_DIRECTORY}"
        )

    track_folders = sorted([
        path
        for path
        in MULTI_LABEL_EXAMPLES_DIRECTORY.iterdir()
        if path.is_dir()
    ])

    if len(
        track_folders
    ) == 0:

        raise ValueError(
            "No track folders were found inside "
            "multi_label_examples."
        )

    return track_folders


@st.cache_data(
    show_spinner=False
)
def process_selected_audio(
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
        len(
            audio
        ),
        clip_length
    ):

        end_sample = (
            start_sample
            + clip_length
        )

        sample = audio[
            start_sample:end_sample
        ]

        if len(
            sample
        ) != clip_length:

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

    pipeline = (
        MultiLabelInferencePipeline(
            checkpoint_path=Path(
                checkpoint_path
            ),
            data_representation=(
                representation_function
            )
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
        figsize=(
            12,
            4
        )
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


# ---------------------------------------------------------------------
# Select example audio
# ---------------------------------------------------------------------

st.title(
    "Multi-Label Audio Classification System"
)

st.write(
    "Select one of the held-out Slakh test examples to inspect "
    "preprocessing and run multi-label Transformer inference."
)

try:

    example_track_folders = (
        get_example_track_folders()
    )

except (
    FileNotFoundError,
    ValueError
) as error:

    st.error(
        str(
            error
        )
    )

    st.stop()


example_options = {}

for (
    index,
    track_folder
) in enumerate(
    example_track_folders,
    start=1
):

    example_name = (
        f"Example {index} | "
        f"{track_folder.name}"
    )

    example_options[
        example_name
    ] = {
        "example_number":
            index,

        "track_folder":
            track_folder,

        "mix_path":
            track_folder
            / "mix.flac"
    }


selected_example_name = (
    st.selectbox(
        "Select Example Audio",
        list(
            example_options.keys()
        )
    )
)

selected_example = (
    example_options[
        selected_example_name
    ]
)

selected_example_number = (
    selected_example[
        "example_number"
    ]
)

selected_track_folder = (
    selected_example[
        "track_folder"
    ]
)

audio_file_path = (
    selected_example[
        "mix_path"
    ]
)

if not audio_file_path.exists():

    st.error(
        "mix.flac could not be found for "
        f"{selected_track_folder.name}: "
        f"{audio_file_path}"
    )

    st.stop()


# Alias used by the inference pipeline to map
# Example 1 -> first track folder, etc.
inference_example_path = Path(
    f"example{selected_example_number}.flac"
)


# ---------------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------------

try:

    (
        audio,
        sample_rate,
        all_samples,
        processing_statistics
    ) = (
        process_selected_audio(
            str(
                audio_file_path
            )
        )
    )

except ValueError as error:

    st.error(
        f"Audio processing failed: "
        f"{error}"
    )

    st.stop()


duration = (
    len(
        audio
    )
    / sample_rate
)


# ---------------------------------------------------------------------
# Audio preview
# ---------------------------------------------------------------------

st.subheader(
    "Audio Preview"
)

preview_col1, preview_col2 = (
    st.columns(
        [
            2,
            1
        ]
    )
)

with preview_col1:

    st.audio(
        audio_to_bytes(
            audio,
            sample_rate
        ),
        format="audio/wav"
    )

with preview_col2:

    st.write(
        f"**Track:** "
        f"{selected_track_folder.name}"
    )

    st.write(
        f"**Duration:** "
        f"{duration:.2f} seconds"
    )

    st.write(
        f"**Sample Rate:** "
        f"{sample_rate} Hz"
    )


# ---------------------------------------------------------------------
# Data processing demo
# ---------------------------------------------------------------------

st.header(
    "Data Processing Demo"
)

st.write(
    "The uploaded audio is divided into 5-second "
    "samples, filtered by loudness, and converted "
    "into the selected spectrogram representation."
)

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
        <div style="
            border:1px solid #cccccc;
            border-radius:8px;
            padding:20px;
            text-align:center;
        ">
            <strong>Audio File</strong>
            <br><br>
            {duration:.2f} seconds
        </div>
        """,
        unsafe_allow_html=True
    )

with arrow_col1:

    st.markdown(
        "<h2 style='text-align:center;'>→</h2>",
        unsafe_allow_html=True
    )

with pipeline_col2:

    st.markdown(
        f"""
        <div style="
            border:1px solid #cccccc;
            border-radius:8px;
            padding:20px;
            text-align:center;
        ">
            <strong>Split Audio</strong>
            <br><br>
            {processing_statistics["total_sub_samples"]} samples
        </div>
        """,
        unsafe_allow_html=True
    )

with arrow_col2:

    st.markdown(
        "<h2 style='text-align:center;'>→</h2>",
        unsafe_allow_html=True
    )

with pipeline_col3:

    st.markdown(
        f"""
        <div style="
            border:1px solid #cccccc;
            border-radius:8px;
            padding:20px;
            text-align:center;
        ">
            <strong>Filtered Audio</strong>
            <br><br>
            {processing_statistics["valid_sub_samples"]} samples
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------
# Processing statistics
# ---------------------------------------------------------------------

st.subheader(
    "Processing Statistics"
)

stats_col1, stats_col2, stats_col3 = (
    st.columns(
        3
    )
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


# ---------------------------------------------------------------------
# Audio sample explorer
# ---------------------------------------------------------------------

st.subheader(
    "Audio Sample Explorer"
)

if len(
    all_samples
) == 0:

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

    if len(
        displayed_samples
    ) == 0:

        st.warning(
            "No samples are available in this category."
        )

    else:

        sample_options = {}

        for (
            index,
            sample
        ) in enumerate(
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
            st.columns(
                2
            )
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
            f"Generating "
            f"{representation_name} spectrogram..."
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


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

st.header(
    "Inference"
)

st.write(
    "Run the trained multi-label Transformer on the "
    "uploaded audio. Multiple instrument classes may be "
    "predicted at the same time."
)

input_col, model_col = (
    st.columns(
        2
    )
)

with input_col:

    st.subheader(
        "Input Audio File"
    )

    st.write(
        f"**{selected_example_name}**"
    )

    st.audio(
        audio_to_bytes(
            audio,
            sample_rate
        ),
        format="audio/wav"
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
        f"**Model:** "
        f"{selected_model_name}"
    )

    st.write(
        "**Input Representation:** "
        f"{selected_model['representation']}"
    )

    st.write(
        "**Task:** "
        "Multi-Label Instrument Classification"
    )

    st.write(
        "**Architecture:** Transformer"
    )


# ---------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------

st.subheader(
    "Run Inference"
)

run_inference = (
    st.button(
        "Run Inference",
        type="primary",
        use_container_width=True
    )
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
                "Running multi-label inference..."
            ):

                results = (
                    inference_pipeline.predict(
                        inference_example_path
                    )
                )

            inference_time = (
                time.perf_counter()
                - start_time
            )

            st.session_state[
                "multi_inference_results"
            ] = results

            st.session_state[
                "multi_inference_time"
            ] = inference_time

            st.session_state[
                "multi_inference_model"
            ] = selected_model_name

            st.session_state[
                "multi_inference_representation"
            ] = representation_name

            st.session_state[
                "multi_inference_example"
            ] = selected_example_name

        except Exception as error:

            st.error(
                f"Inference failed: "
                f"{error}"
            )


# ---------------------------------------------------------------------
# Inference results
# ---------------------------------------------------------------------

if (
    "multi_inference_results"
    in st.session_state
    and st.session_state.get(
        "multi_inference_example"
    )
    == selected_example_name
):

    results = (
        st.session_state[
            "multi_inference_results"
        ]
    )

    st.subheader(
        "Inference Results"
    )

    predicted_labels = (
        results[
            "predicted_labels"
        ]
    )

    threshold = (
        results[
            "classification_threshold"
        ]
    )

    if len(
        predicted_labels
    ) == 0:

        st.warning(
            "No instrument classes exceeded the "
            f"{threshold:.2f} classification threshold."
        )

    else:

        st.write(
            "**Predicted Instruments:**"
        )

        prediction_columns = (
            st.columns(
                min(
                    len(
                        predicted_labels
                    ),
                    4
                )
            )
        )

        for (
            index,
            label
        ) in enumerate(
            predicted_labels
        ):

            probability = (
                results[
                    "class_probabilities"
                ][
                    label
                ]
            )

            with prediction_columns[
                index
                % len(
                    prediction_columns
                )
            ]:

                st.metric(
                    label,
                    (
                        f"{probability * 100:.2f}%"
                    )
                )

    st.caption(
        f"Classification threshold: "
        f"{threshold:.2f}"
    )


    # -----------------------------------------------------------------
    # Class probabilities
    # -----------------------------------------------------------------

    st.subheader(
        "Class Probabilities"
    )

    probability_dataframe = (
        pd.DataFrame(
            [
                {
                    "Instrument":
                        label,

                    "Probability":
                        probability
                        * 100,

                    "Predicted":
                        (
                            "Yes"
                            if probability
                            >= threshold
                            else "No"
                        )
                }

                for (
                    label,
                    probability
                ) in results[
                    "class_probabilities"
                ].items()
            ]
        )
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


    # -----------------------------------------------------------------
    # Additional information
    # -----------------------------------------------------------------

    st.subheader(
        "Additional Information"
    )

    info_col1, info_col2, info_col3, info_col4 = (
        st.columns(
            4
        )
    )

    with info_col1:

        st.write(
            "**Model Used**"
        )

        st.write(
            st.session_state[
                "multi_inference_model"
            ]
        )

    with info_col2:

        st.write(
            "**Input Representation**"
        )

        st.write(
            st.session_state[
                "multi_inference_representation"
            ]
        )

    with info_col3:

        st.write(
            "**Inference Time**"
        )

        st.write(
            f"{st.session_state['multi_inference_time']:.3f} "
            "seconds"
        )


    with info_col4:

        st.write(
            "**Example Track**"
        )

        st.write(
            selected_track_folder.name
        )


    # -----------------------------------------------------------------
    # Per-sample predictions
    # -----------------------------------------------------------------

    with st.expander(
        "5-Second Sample Predictions"
    ):

        sample_rows = []

        for sample_result in (
            results[
                "sample_predictions"
            ]
        ):

            predicted_labels = (
                sample_result[
                    "predicted_labels"
                ]
            )

            true_labels = (
                sample_result.get(
                    "true_labels"
                )
            )

            if len(
                predicted_labels
            ) == 0:

                predicted_text = (
                    "No active labels"
                )

            else:

                predicted_text = (
                    ", ".join(
                        predicted_labels
                    )
                )

            if true_labels is None:

                true_text = (
                    "Not available"
                )

            elif len(
                true_labels
            ) == 0:

                true_text = (
                    "No active labels"
                )

            else:

                true_text = (
                    ", ".join(
                        true_labels
                    )
                )

            row = {
                "Sample":
                    sample_result[
                        "sample"
                    ],

                "True Labels":
                    true_text,

                "Predicted Instruments":
                    predicted_text
            }

            if (
                sample_result.get(
                    "start_time"
                )
                is not None
                and sample_result.get(
                    "end_time"
                )
                is not None
            ):

                row[
                    "Time"
                ] = (
                    f"{sample_result['start_time']:.1f}s - "
                    f"{sample_result['end_time']:.1f}s"
                )

            sample_rows.append(
                row
            )

        sample_dataframe = (
            pd.DataFrame(
                sample_rows
            )
        )

        if (
            results.get(
                "true_labels_available",
                False
            )
        ):

            st.caption(
                "True labels are generated from the corresponding "
                "Slakh track stems in multi_label_examples."
            )

        else:

            st.caption(
                "True labels are only available for example1 "
                "through example5."
            )

        st.dataframe(
            sample_dataframe,
            use_container_width=True,
            hide_index=True
        )