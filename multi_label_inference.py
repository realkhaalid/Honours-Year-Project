from pathlib import Path
import re

import torch

from multi_label_transformer_class import (
    MultiLabelTransformer
)

from data_processing_pipeline import (
    load_and_validate_audio,
    split_audio_into_subsamples,
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram
)

from multi_label_data_processing import (
    process_multilabel_track,
    load_multilabel_mix_sample
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_REPRESENTATION = (
    convert_to_mel_spectrogram
)

CHECKPOINT_PATH = Path(
    "multi_label_checkpoints/"
    "mel_multi_label_pretrained_fine_tuned_supervised_best.pt"
)

MULTI_LABEL_EXAMPLES_DIRECTORY = Path(
    "multi_label_examples"
)


# ---------------------------------------------------------------------
# Multi-label inference pipeline
# ---------------------------------------------------------------------

class MultiLabelInferencePipeline:

    def __init__(
        self,
        checkpoint_path,
        data_representation=(
            convert_to_mel_spectrogram
        ),
        examples_directory=(
            MULTI_LABEL_EXAMPLES_DIRECTORY
        )
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.data_representation = (
            data_representation
        )

        self.examples_directory = Path(
            examples_directory
        )

        self.model = None

        self.label_to_index = None
        self.index_to_label = None

        self.classification_threshold = None

        print(
            "Inference device:",
            self.device
        )


    # -----------------------------------------------------------------
    # Load trained model
    # -----------------------------------------------------------------

    def load_model(
        self,
        sample_batch
    ):

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if (
            checkpoint.get(
                "phase"
            )
            != "supervised"
        ):

            raise ValueError(
                "Inference requires a supervised "
                "checkpoint."
            )

        checkpoint_type = (
            checkpoint.get(
                "checkpoint_type"
            )
        )

        if checkpoint_type not in (
            "latest",
            "best"
        ):

            raise ValueError(
                f"Unsupported checkpoint type: "
                f"{checkpoint_type}"
            )

        configuration = (
            checkpoint[
                "model_configuration"
            ]
        )

        sample_batch = sample_batch.to(
            device=self.device,
            dtype=torch.float32
        )

        if (
            sample_batch.shape[-1]
            != configuration[
                "patch_dim"
            ]
        ):

            raise ValueError(
                "Input patch dimension does not "
                "match checkpoint. "
                f"Input: "
                f"{sample_batch.shape[-1]}, "
                f"checkpoint: "
                f"{configuration['patch_dim']}."
            )

        self.model = (
            MultiLabelTransformer(
                sample_batch=(
                    sample_batch
                ),
                embedding_dim=(
                    configuration[
                        "embedding_dim"
                    ]
                ),
                num_heads=(
                    configuration[
                        "num_heads"
                    ]
                ),
                hidden_dims=tuple(
                    configuration[
                        "hidden_dims"
                    ]
                ),
                num_encoder_layers=(
                    configuration[
                        "num_encoder_layers"
                    ]
                ),
                num_classes=(
                    configuration[
                        "num_classes"
                    ]
                ),
                mask_ratio=(
                    configuration[
                        "mask_ratio"
                    ]
                ),
                activation=(
                    configuration[
                        "activation"
                    ]
                ),
                classification_threshold=(
                    configuration.get(
                        "classification_threshold",
                        0.5
                    )
                )
            )
        )

        if checkpoint_type == "best":

            model_state = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model_state = (
                checkpoint.get(
                    "best_model_state_dict"
                )
            )

            if model_state is None:

                model_state = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

        self.model.load_state_dict(
            model_state
        )

        self.label_to_index = (
            checkpoint[
                "label_to_index"
            ]
        )

        self.index_to_label = (
            checkpoint[
                "index_to_label"
            ]
        )

        self.classification_threshold = (
            configuration.get(
                "classification_threshold",
                0.5
            )
        )

        print(
            "Model loaded successfully."
        )

        print(
            "Number of classes:",
            len(
                self.label_to_index
            )
        )

        print(
            "Classification threshold:",
            self.classification_threshold
        )


    # -----------------------------------------------------------------
    # Example folder helpers
    # -----------------------------------------------------------------

    def _get_example_track_folders(
        self
    ):

        if not self.examples_directory.exists():

            return []

        track_folders = sorted([
            path
            for path in (
                self.examples_directory.iterdir()
            )
            if path.is_dir()
        ])

        return track_folders


    def _get_example_number(
        self,
        audio_path
    ):

        audio_path = Path(
            audio_path
        )

        match = re.search(
            r"example[\s_-]*(\d+)",
            audio_path.stem,
            flags=re.IGNORECASE
        )

        if match is None:

            return None

        return int(
            match.group(
                1
            )
        )


    def _resolve_example_track_folder(
        self,
        audio_path
    ):
        """
        Maps example1 -> first sorted track folder,
        example2 -> second sorted track folder, etc.
        """

        example_number = (
            self._get_example_number(
                audio_path
            )
        )

        if example_number is None:

            return None

        track_folders = (
            self._get_example_track_folders()
        )

        if len(
            track_folders
        ) == 0:

            raise FileNotFoundError(
                "No track folders were found in "
                f"{self.examples_directory}."
            )

        example_index = (
            example_number
            - 1
        )

        if (
            example_index < 0
            or example_index
            >= len(
                track_folders
            )
        ):

            raise ValueError(
                f"example{example_number} does not "
                "have a matching track folder in "
                f"{self.examples_directory}. "
                f"Found {len(track_folders)} folders."
            )

        return (
            track_folders[
                example_index
            ]
        )


    # -----------------------------------------------------------------
    # Standard uploaded-audio processing
    # -----------------------------------------------------------------

    def _process_standard_audio(
        self,
        audio_path
    ):

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_path}"
            )

        audio, sample_rate = (
            load_and_validate_audio(
                audio_path
            )
        )

        (
            valid_patches,
            statistics
        ) = (
            split_audio_into_subsamples(
                audio,
                sample_rate,
                data_representation=(
                    self.data_representation
                )
            )
        )

        if len(
            valid_patches
        ) == 0:

            raise ValueError(
                "No valid 5-second audio samples "
                "were found."
            )

        samples = []

        for (
            sample_number,
            patches
        ) in enumerate(
            valid_patches,
            start=1
        ):

            samples.append({
                "sample":
                    sample_number,

                "patches":
                    patches,

                "true_labels":
                    None,

                "start_time":
                    None,

                "end_time":
                    None
            })

        return (
            samples,
            statistics,
            None
        )


    # -----------------------------------------------------------------
    # Slakh example processing with true labels
    # -----------------------------------------------------------------

    def _process_example_track(
        self,
        audio_path
    ):

        track_folder = (
            self._resolve_example_track_folder(
                audio_path
            )
        )

        if track_folder is None:

            return None

        (
            sample_metadata,
            statistics
        ) = (
            process_multilabel_track(
                track_folder=(
                    track_folder
                )
            )
        )

        if len(
            sample_metadata
        ) == 0:

            raise ValueError(
                "The selected example track has "
                "no valid multi-label samples."
            )

        samples = []

        for (
            sample_number,
            sample_information
        ) in enumerate(
            sample_metadata,
            start=1
        ):

            (
                patches,
                label_vector
            ) = (
                load_multilabel_mix_sample(
                    sample_information=(
                        sample_information
                    ),
                    data_representation=(
                        self.data_representation
                    )
                )
            )

            samples.append({
                "sample":
                    sample_number,

                "patches":
                    patches,

                "true_labels":
                    list(
                        sample_information[
                            "labels"
                        ]
                    ),

                "true_label_indices":
                    list(
                        sample_information[
                            "label_indices"
                        ]
                    ),

                "start_time":
                    sample_information[
                        "start_time"
                    ],

                "end_time":
                    sample_information[
                        "end_time"
                    ]
            })

        return (
            samples,
            statistics,
            track_folder
        )


    # -----------------------------------------------------------------
    # Process audio
    # -----------------------------------------------------------------

    def process_audio(
        self,
        audio_path
    ):
        """
        For files named example1, example2, ... the
        corresponding Slakh folder inside
        multi_label_examples is used to obtain the
        ground-truth labels.

        Other audio files are processed normally and
        have no true labels.
        """

        example_result = (
            self._process_example_track(
                audio_path
            )
        )

        if example_result is not None:

            return example_result

        return (
            self._process_standard_audio(
                audio_path
            )
        )


    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _indices_to_labels(
        self,
        prediction_vector
    ):

        predicted_indices = (
            torch.nonzero(
                prediction_vector,
                as_tuple=False
            )
            .flatten()
            .tolist()
        )

        predicted_labels = [
            self.index_to_label[
                index
            ]
            for index
            in predicted_indices
        ]

        return (
            predicted_indices,
            predicted_labels
        )


    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict(
        self,
        audio_path
    ):

        (
            processed_samples,
            statistics,
            example_track_folder
        ) = (
            self.process_audio(
                audio_path
            )
        )

        print(
            "Total 5-second samples:",
            statistics[
                "total_sub_samples"
            ]
        )

        print(
            "Valid samples:",
            statistics[
                "valid_sub_samples"
            ]
        )

        if (
            "quiet_sub_samples"
            in statistics
        ):

            quiet_samples = (
                statistics[
                    "quiet_sub_samples"
                ]
            )

        else:

            quiet_samples = (
                statistics.get(
                    "quiet_mix_sub_samples",
                    0
                )
            )

        print(
            "Quiet samples removed:",
            quiet_samples
        )

        sample_batch = (
            processed_samples[
                0
            ][
                "patches"
            ]
            .to(
                dtype=torch.float32
            )
        )

        if sample_batch.ndim == 2:

            sample_batch = (
                sample_batch.unsqueeze(
                    0
                )
            )

        if self.model is None:

            self.load_model(
                sample_batch
            )

        sample_probabilities = []
        sample_predictions = []

        with torch.no_grad():

            for sample_information in (
                processed_samples
            ):

                patches = (
                    sample_information[
                        "patches"
                    ]
                    .to(
                        dtype=torch.float32,
                        device=self.device
                    )
                )

                if patches.ndim == 2:

                    patches = (
                    patches.unsqueeze(
                        0
                    )
                )

                (
                    predicted_vector,
                    probabilities
                ) = (
                    self.model.classify(
                        patches
                    )
                )

                sample_probabilities.append(
                    probabilities
                    .detach()
                    .cpu()
                )

                predicted_vector_cpu = (
                    predicted_vector[
                        0
                    ]
                    .detach()
                    .cpu()
                    .to(
                        dtype=torch.bool
                    )
                )

                (
                    predicted_indices,
                    predicted_labels
                ) = (
                    self._indices_to_labels(
                        predicted_vector_cpu
                    )
                )

                class_probabilities = {}

                for index in range(
                    probabilities.shape[-1]
                ):

                    label = (
                        self.index_to_label[
                            index
                        ]
                    )

                    probability = (
                        probabilities[
                            0,
                            index
                        ]
                        .detach()
                        .cpu()
                        .item()
                    )

                    class_probabilities[
                        label
                    ] = probability

                sample_result = {
                    "sample":
                        sample_information[
                            "sample"
                        ],

                    "predicted_indices":
                        predicted_indices,

                    "predicted_labels":
                        predicted_labels,

                    "true_labels":
                        sample_information.get(
                            "true_labels"
                        ),

                    "start_time":
                        sample_information.get(
                            "start_time"
                        ),

                    "end_time":
                        sample_information.get(
                            "end_time"
                        ),

                    "class_probabilities":
                        class_probabilities
                }

                sample_predictions.append(
                    sample_result
                )

        all_probabilities = torch.cat(
            sample_probabilities,
            dim=0
        )

        average_probabilities = (
            torch.mean(
                all_probabilities,
                dim=0
            )
        )

        final_prediction_vector = (
            average_probabilities
            >= self.classification_threshold
        )

        (
            predicted_indices,
            predicted_labels
        ) = (
            self._indices_to_labels(
                final_prediction_vector
            )
        )

        class_probabilities = {}

        for index in range(
            average_probabilities.shape[0]
        ):

            label = (
                self.index_to_label[
                    index
                ]
            )

            class_probabilities[
                label
            ] = (
                average_probabilities[
                    index
                ]
                .item()
            )

        true_labels_available = (
            example_track_folder
            is not None
        )

        return {
            "predicted_indices":
                predicted_indices,

            "predicted_labels":
                predicted_labels,

            "class_probabilities":
                class_probabilities,

            "sample_predictions":
                sample_predictions,

            "processing_statistics":
                statistics,

            "classification_threshold":
                self.classification_threshold,

            "true_labels_available":
                true_labels_available,

            "example_track_folder":
                (
                    str(
                        example_track_folder
                    )
                    if example_track_folder
                    is not None
                    else None
                )
        }


# ---------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------

if __name__ == "__main__":

    inference_pipeline = (
        MultiLabelInferencePipeline(
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            data_representation=(
                DATA_REPRESENTATION
            ),
            examples_directory=(
                MULTI_LABEL_EXAMPLES_DIRECTORY
            )
        )
    )

    AUDIO_FILE = Path(
        "example1.flac"
    )

    results = (
        inference_pipeline.predict(
            AUDIO_FILE
        )
    )

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "MULTI-LABEL INFERENCE RESULTS"
    )

    print(
        "=" * 70
    )

    if (
        results[
            "example_track_folder"
        ]
        is not None
    ):

        print(
            "Ground-truth track folder:",
            results[
                "example_track_folder"
            ]
        )

    print(
        "\nPredicted instruments:"
    )

    if len(
        results[
            "predicted_labels"
        ]
    ) == 0:

        print(
            "No instruments exceeded "
            "the classification threshold."
        )

    else:

        for label in (
            results[
                "predicted_labels"
            ]
        ):

            probability = (
                results[
                    "class_probabilities"
                ][
                    label
                ]
            )

            print(
                f"{label:<30} "
                f"{probability * 100:>7.2f}%"
            )

    print(
        "\n5-Second Sample Predictions"
    )

    print(
        "-" * 70
    )

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
            sample_result[
                "true_labels"
            ]
        )

        predicted_text = (
            ", ".join(
                predicted_labels
            )
            if len(
                predicted_labels
            ) > 0
            else "No active labels"
        )

        if true_labels is None:

            true_text = (
                "Not available"
            )

        else:

            true_text = (
                ", ".join(
                    true_labels
                )
                if len(
                    true_labels
                ) > 0
                else "No active labels"
            )

        print(
            f"Sample "
            f"{sample_result['sample']:>3}"
        )

        print(
            f"  True:      "
            f"{true_text}"
        )

        print(
            f"  Predicted: "
            f"{predicted_text}"
        )