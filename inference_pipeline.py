from pathlib import Path

import torch

from transformer_model_class import Transformer

from data_processing_pipeline import (
    load_and_validate_audio,
    split_audio_into_subsamples,
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram
)

# Configuration
DATA_REPRESENTATION = convert_to_mel_spectrogram

CHECKPOINT_PATH = Path(
    "checkpoints/mel_pretrained_fine_tuned_test_supervised_latest.pt"
)

# Inference Pipeline
class InferencePipeline:

    def __init__(
        self,
        checkpoint_path,
        data_representation=convert_to_mel_spectrogram
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

        self.model = None

        self.label_to_index = None
        self.index_to_label = None

        print(
            "Inference device:",
            self.device
        )

    # Load model
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

        checkpoint_type = checkpoint.get(
        "checkpoint_type"
        )

        if checkpoint_type not in (
            "latest",
            "best"
        ):

            raise ValueError(
                f"Unsupported checkpoint type: "
                f"{checkpoint_type}"
            )

        configuration = checkpoint[
            "model_configuration"
        ]

        self.model = Transformer(
            sample_batch=sample_batch.to(
                self.device
            ),
            embedding_dim=configuration[
                "embedding_dim"
            ],
            num_heads=configuration[
                "num_heads"
            ],
            hidden_dims=tuple(
                configuration[
                    "hidden_dims"
                ]
            ),
            num_encoder_layers=configuration[
                "num_encoder_layers"
            ],
            num_classes=configuration[
                "num_classes"
            ],
            mask_ratio=configuration[
                "mask_ratio"
            ],
            activation=configuration[
                "activation"
            ]
        )

        if checkpoint_type == "best":

            model_state = checkpoint[
                "model_state_dict"
            ]

        else:

            model_state = checkpoint.get(
                "best_model_state_dict"
            )

        if model_state == None:
            raise ValueError(
                "Unable to load model state"
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.label_to_index = checkpoint[
            "label_to_index"
        ]

        self.index_to_label = checkpoint[
            "index_to_label"
        ]

        print(
            "Model loaded successfully."
        )

        print(
            "Number of classes:",
            len(
                self.label_to_index
            )
        )

    # Process audio using training preprocessing pipeline
    def process_audio(
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

        if len(valid_patches) == 0:

            raise ValueError(
                "No valid 5-second audio samples "
                "were found."
            )

        return (
            valid_patches,
            statistics
        )
    
    # Predict audio
    def predict(
        self,
        audio_path
    ):

        (
            valid_patches,
            statistics
        ) = self.process_audio(
            audio_path
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

        print(
            "Quiet samples removed:",
            statistics[
                "quiet_sub_samples"
            ]
        )

        sample_batch = (
            valid_patches[0]
            .to(
                dtype=torch.float32
            )
        )

        if self.model is None:

            self.load_model(
                sample_batch
            )

        sample_probabilities = []
        sample_predictions = []

        with torch.no_grad():

            for sample_number, patches in enumerate(
                valid_patches,
                start=1
            ):

                patches = (
                    patches
                    .to(
                        dtype=torch.float32,
                        device=self.device
                    )
                )

                (
                    predicted_index,
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

                predicted_index = (
                    predicted_index
                    .item()
                )

                predicted_label = (
                    self.index_to_label[
                        predicted_index
                    ]
                )

                confidence = (
                    probabilities[
                        0,
                        predicted_index
                    ]
                    .item()
                )

                sample_predictions.append({
                    "sample":
                        sample_number,

                    "predicted_index":
                        predicted_index,

                    "predicted_label":
                        predicted_label,

                    "confidence":
                        confidence
                })

        all_probabilities = torch.cat(
            sample_probabilities,
            dim=0
        )

        average_probabilities = torch.mean(
            all_probabilities,
            dim=0,
            keepdim=True
        )

        predicted_index = (
            torch.argmax(
                average_probabilities,
                dim=-1
            )
            .item()
        )

        predicted_label = (
            self.index_to_label[
                predicted_index
            ]
        )

        confidence = (
            average_probabilities[
                0,
                predicted_index
            ]
            .item()
        )

        class_probabilities = {}

        for index in range(
            average_probabilities.shape[-1]
        ):

            label = (
                self.index_to_label[
                    index
                ]
            )

            probability = (
                average_probabilities[
                    0,
                    index
                ]
                .item()
            )

            class_probabilities[
                label
            ] = probability

        return {
            "predicted_index":
                predicted_index,

            "predicted_label":
                predicted_label,

            "confidence":
                confidence,

            "class_probabilities":
                class_probabilities,

            "sample_predictions":
                sample_predictions,

            "processing_statistics":
                statistics
        }

# Run inference
if __name__ == "__main__":

    inference_pipeline = (
        InferencePipeline(
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            data_representation=(
                DATA_REPRESENTATION
            )
        )
    )

    AUDIO_FILE = Path(
        "audio_examples/S00.flac"
    )

    results = (
        inference_pipeline.predict(
            AUDIO_FILE
        )
    )

    print("\n")
    print("=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)

    print(
        "Predicted instrument:",
        results[
            "predicted_label"
        ]
    )

    print(
        "Predicted class index:",
        results[
            "predicted_index"
        ]
    )

    print(
        "Confidence:",
        f"{results['confidence'] * 100:.2f}%"
    )

    print(
        "\nClass Probabilities"
    )

    print("-" * 70)

    sorted_probabilities = sorted(
        results[
            "class_probabilities"
        ].items(),
        key=lambda item: item[1],
        reverse=True
    )

    for label, probability in (
        sorted_probabilities
    ):

        print(
            f"{label:<30} "
            f"{probability * 100:>7.2f}%"
        )

    print(
        "\n5-Second Sample Predictions"
    )

    print("-" * 70)

    for sample_result in (
        results[
            "sample_predictions"
        ]
    ):

        print(
            f"Sample "
            f"{sample_result['sample']:>3} | "
            f"{sample_result['predicted_label']:<25} | "
            f"{sample_result['confidence'] * 100:>7.2f}%"
        )