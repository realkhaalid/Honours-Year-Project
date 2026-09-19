import torch

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformer_model_class import Transformer

from data_processing_pipeline import (
    convert_to_mel_spectrogram
)

from construct_dataset_class import (
    create_test_dataset,
    SLAKH2100_REDUX_16K_TEST
)


# Config
BATCH_SIZE = 16

DATA_REPRESENTATION = (
    convert_to_mel_spectrogram
)

CHECKPOINT_PATH = Path(
    "checkpoints/mel_supervised_only_test_supervised_best.pt"
)

class TestModelPipeline:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Testing device:",
            self.device
        )

        if torch.cuda.is_available():
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

    def load_latest_model(
        self,
        checkpoint_path,
        sample_batch
    ):
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        print(
            "\nCheckpoint loaded:",
            checkpoint_path
        )

        print(
            "Checkpoint type:",
            checkpoint.get("checkpoint_type")
        )

        print(
            "Phase:",
            checkpoint.get("phase")
        )

        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch")
        )

        print(
            "Best epoch:",
            checkpoint.get("best_epoch")
        )

        print(
            "Best validation loss:",
            checkpoint.get(
                "best_validation_loss"
            )
        )

        if checkpoint.get("phase") != "supervised":
            raise ValueError(
                "The checkpoint must be from "
                "the supervised phase."
            )

        model_configuration = checkpoint[
            "model_configuration"
        ]

        sample_batch = sample_batch.to(
            self.device
        )

        model = Transformer(
            sample_batch=sample_batch,
            embedding_dim=model_configuration[
                "embedding_dim"
            ],
            num_heads=model_configuration[
                "num_heads"
            ],
            hidden_dims=model_configuration[
                "hidden_dims"
            ],
            num_encoder_layers=model_configuration[
                "num_encoder_layers"
            ],
            num_classes=model_configuration[
                "num_classes"
            ],
            mask_ratio=model_configuration[
                "mask_ratio"
            ],
            activation=model_configuration[
                "activation"
            ]
        )

        sample_patch_dimension = (
            sample_batch.shape[-1]
        )

        print(
            "Sample patch dimension:",
            sample_patch_dimension
        )

        if (
            checkpoint.get(
                "best_model_state_dict"
            )
            is None
        ):
            raise ValueError(
                "No best model state was stored "
                "inside the latest checkpoint."
            )

        model.load_state_dict(
            checkpoint[
                "best_model_state_dict"
            ]
        )

        print(
            "Best model state loaded from "
            "latest checkpoint."
        )

        return (
            model,
            checkpoint
        )

    def load_best_model(
        self,
        checkpoint_path,
        sample_batch
    ):
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        print(
            "\nCheckpoint loaded:",
            checkpoint_path
        )

        print(
            "Checkpoint type:",
            checkpoint.get("checkpoint_type")
        )

        print(
            "Phase:",
            checkpoint.get("phase")
        )

        print(
            "Best epoch:",
            checkpoint.get("best_epoch")
        )

        print(
            "Best validation loss:",
            checkpoint.get(
                "best_validation_loss"
            )
        )

        if (
            checkpoint.get("checkpoint_type")
            != "best"
        ):
            raise ValueError(
                "The checkpoint must be "
                "a best checkpoint."
            )

        if checkpoint.get("phase") != "supervised":
            raise ValueError(
                "The checkpoint must be from "
                "the supervised phase."
            )

        model_configuration = checkpoint[
            "model_configuration"
        ]

        sample_batch = sample_batch.to(
            self.device
        )

        model = Transformer(
            sample_batch=sample_batch,
            embedding_dim=model_configuration[
                "embedding_dim"
            ],
            num_heads=model_configuration[
                "num_heads"
            ],
            hidden_dims=model_configuration[
                "hidden_dims"
            ],
            num_encoder_layers=model_configuration[
                "num_encoder_layers"
            ],
            num_classes=model_configuration[
                "num_classes"
            ],
            mask_ratio=model_configuration[
                "mask_ratio"
            ],
            activation=model_configuration[
                "activation"
            ]
        )

        sample_patch_dimension = (
            sample_batch.shape[-1]
        )

        print(
            "Sample patch dimension:",
            sample_patch_dimension
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Best checkpoint model loaded "
            "successfully."
        )

        return (
            model,
            checkpoint
        )


    def test_model(
        self,
        model,
        test_loader,
        index_to_label
    ):

        total_loss = 0.0
        number_of_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for (
                patches,
                labels
            ) in tqdm(test_loader):

                patches = patches.to(
                    self.device,
                    dtype=torch.float32
                )

                labels = labels.to(
                    self.device,
                    dtype=torch.long
                )

                (
                    loss,
                    logits,
                    predicted_classes,
                    probabilities,
                    attention_weights
                ) = (
                    model
                    .supervised_fine_tuning(
                        patches,
                        labels
                    )
                )

                if not torch.isfinite(
                    loss
                ):
                    raise ValueError(
                        "Non-finite test loss "
                        "detected."
                    )

                predictions = (
                    predicted_classes
                )

                total_loss += (
                    loss.item()
                )

                number_of_batches += 1

                all_predictions.extend(
                    predictions
                    .detach()
                    .cpu()
                    .tolist()
                )

                all_labels.extend(
                    labels
                    .detach()
                    .cpu()
                    .tolist()
                )

        if number_of_batches == 0:

            raise ValueError(
                "The test DataLoader "
                "contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )

        class_indices = sorted(
            index_to_label.keys()
        )

        class_names = [
            index_to_label[index]
            for index
            in class_indices
        ]

        report = classification_report(
            all_labels,
            all_predictions,
            labels=class_indices,
            target_names=class_names,
            zero_division=0
        )

        report_dictionary = (
            classification_report(
                all_labels,
                all_predictions,
                labels=class_indices,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )
        )

        matrix = confusion_matrix(
            all_labels,
            all_predictions,
            labels=class_indices
        )

        return {
            "test_loss":
                average_loss,

            "accuracy":
                accuracy,

            "macro_precision":
                macro_precision,

            "macro_recall":
                macro_recall,

            "macro_f1":
                macro_f1,

            "weighted_precision":
                weighted_precision,

            "weighted_recall":
                weighted_recall,

            "weighted_f1":
                weighted_f1,

            "classification_report":
                report,

            "classification_report_dictionary":
                report_dictionary,

            "confusion_matrix":
                matrix
        }

if __name__ == "__main__":

    # Create test dataset
    print("Creating test dataset...")

    test_dataset = create_test_dataset(
        test_path=SLAKH2100_REDUX_16K_TEST,
        data_representation=DATA_REPRESENTATION,
        set_limit=False
    )

    (
        test_label_to_index,
        test_index_to_label
    ) = test_dataset.return_label_mappings()

    print(
        "Test dataset size:",
        len(test_dataset)
    )

    print(
        "Test label mapping:",
        test_label_to_index
    )

    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Get one sample batch
    (
        sample_batch,
        _
    ) = next(
        iter(test_loader)
    )

    # Create testing pipeline
    testing_pipeline = TestModelPipeline()

    # Load pretrained + fine-tuned model
    (
        model,
        checkpoint
    ) = testing_pipeline.load_best_model(
        checkpoint_path=CHECKPOINT_PATH,
        sample_batch=sample_batch
    )

    # Check label mappings
    checkpoint_label_to_index = (
        checkpoint[
            "label_to_index"
        ]
    )

    checkpoint_index_to_label = (
        checkpoint[
            "index_to_label"
        ]
    )

    print(
        "\nCheckpoint mapping matches "
        "test mapping:",
        checkpoint_label_to_index
        == test_label_to_index
    )

    if (
        checkpoint_label_to_index
        != test_label_to_index
    ):

        raise ValueError(
            "Checkpoint and test dataset "
            "label mappings do not match."
        )

    # Test model
    print("\nTesting model...")

    test_results = (
        testing_pipeline.test_model(
            model=model,
            test_loader=test_loader,
            index_to_label=(
                checkpoint_index_to_label
            )
        )
    )

    # Print results
    print("\n")
    print("#" * 70)
    print("TEST RESULTS")
    print("#" * 70)

    print(
        f"Test loss: "
        f"{test_results['test_loss']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro precision: "
        f"{test_results['macro_precision']:.4f}"
    )

    print(
        f"Macro recall: "
        f"{test_results['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_results['macro_f1']:.4f}"
    )

    print(
        f"Weighted precision: "
        f"{test_results['weighted_precision']:.4f}"
    )

    print(
        f"Weighted recall: "
        f"{test_results['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_results['weighted_f1']:.4f}"
    )

    print(
        "\nClassification Report"
    )

    print("=" * 70)

    print(
        test_results[
            "classification_report"
        ]
    )

    print(
        "\nConfusion Matrix"
    )

    print("=" * 70)

    print(
        test_results[
            "confusion_matrix"
        ]
    )