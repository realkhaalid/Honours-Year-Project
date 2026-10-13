import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    multilabel_confusion_matrix,
)

from multi_label_transformer_class import MultiLabelTransformer
from data_processing_pipeline import (
    convert_to_mel_spectrogram,
    SLAKH2100_REDUX_16K_TEST,
)
from construct_multi_label_dataset import (
    create_test_dataset,
    MULTI_LABEL_MAPPING,
    INDEX_TO_MULTI_LABEL,
)


BATCH_SIZE = 16
DATA_REPRESENTATION = convert_to_mel_spectrogram
CHECKPOINT_PATH = Path(
    "multi_label_checkpoints/"
    "mel_multi_label_pretrained_fine_tuned_supervised_best.pt"
)


class TestMultiModelPipeline:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Testing device:", self.device)

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

    def load_best_model(self, checkpoint_path, sample_batch):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        print("\nCheckpoint loaded:", checkpoint_path)
        print("Checkpoint type:", checkpoint.get("checkpoint_type"))
        print("Phase:", checkpoint.get("phase"))
        print("Best epoch:", checkpoint.get("best_epoch"))
        print(
            "Best validation loss:",
            checkpoint.get("best_validation_loss"),
        )
        print(
            "Best validation accuracy:",
            checkpoint.get("best_validation_accuracy"),
        )

        if checkpoint.get("checkpoint_type") != "best":
            raise ValueError(
                "The checkpoint must be a best checkpoint."
            )

        if checkpoint.get("phase") != "supervised":
            raise ValueError(
                "The checkpoint must be from the supervised phase."
            )

        configuration = checkpoint["model_configuration"]

        sample_batch = sample_batch.to(
            device=self.device,
            dtype=torch.float32,
        )

        if sample_batch.shape[-1] != configuration["patch_dim"]:
            raise ValueError(
                "Test dataset patch dimension does not match checkpoint. "
                f"Dataset: {sample_batch.shape[-1]}, "
                f"checkpoint: {configuration['patch_dim']}."
            )

        model = MultiLabelTransformer(
            sample_batch=sample_batch,
            embedding_dim=configuration["embedding_dim"],
            num_heads=configuration["num_heads"],
            hidden_dims=tuple(configuration["hidden_dims"]),
            num_encoder_layers=configuration["num_encoder_layers"],
            num_classes=configuration["num_classes"],
            mask_ratio=configuration["mask_ratio"],
            activation=configuration["activation"],
            classification_threshold=configuration.get(
                "classification_threshold",
                0.5,
            ),
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("Best checkpoint model loaded successfully.")

        return model, checkpoint

    def test_model(self, model, test_loader, index_to_label):
        total_loss = 0.0
        number_of_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for patches, labels in tqdm(
                test_loader,
                desc="Testing",
            ):
                patches = patches.to(
                    device=self.device,
                    dtype=torch.float32,
                )

                labels = labels.to(
                    device=self.device,
                    dtype=torch.float32,
                )

                (
                    loss,
                    logits,
                    predicted_labels,
                    probabilities,
                    attention_weights,
                ) = model.supervised_fine_tuning(
                    patches,
                    labels,
                )

                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite test loss detected."
                    )

                total_loss += loss.item()
                number_of_batches += 1

                all_predictions.extend(
                    predicted_labels.detach()
                    .cpu()
                    .to(dtype=torch.int64)
                    .tolist()
                )

                all_labels.extend(
                    labels.detach()
                    .cpu()
                    .to(dtype=torch.int64)
                    .tolist()
                )

        if number_of_batches == 0:
            raise ValueError(
                "The test DataLoader contains no batches."
            )

        average_loss = total_loss / number_of_batches

        accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _,
        ) = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )

        class_indices = sorted(index_to_label.keys())
        class_names = [
            index_to_label[index]
            for index in class_indices
        ]

        report = classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0,
        )

        matrices = multilabel_confusion_matrix(
            all_labels,
            all_predictions,
        )

        return {
            "test_loss": average_loss,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "classification_report": report,
            "confusion_matrices": matrices,
        }


if __name__ == "__main__":
    print("Creating multi-label test dataset...")

    test_dataset = create_test_dataset(
        test_path=SLAKH2100_REDUX_16K_TEST,
        data_representation=DATA_REPRESENTATION,
        set_limit=False,
    )

    (
        test_label_to_index,
        test_index_to_label,
    ) = test_dataset.return_label_mappings()

    if test_label_to_index != MULTI_LABEL_MAPPING:
        raise ValueError(
            "Test dataset mapping does not match MULTI_LABEL_MAPPING."
        )

    if test_index_to_label != INDEX_TO_MULTI_LABEL:
        raise ValueError(
            "Test index mapping does not match INDEX_TO_MULTI_LABEL."
        )

    print("Test dataset size:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    sample_batch, _ = next(iter(test_loader))

    testing_pipeline = TestMultiModelPipeline()

    model, checkpoint = testing_pipeline.load_best_model(
        checkpoint_path=CHECKPOINT_PATH,
        sample_batch=sample_batch,
    )

    checkpoint_label_to_index = checkpoint["label_to_index"]
    checkpoint_index_to_label = checkpoint["index_to_label"]

    if checkpoint_label_to_index != test_label_to_index:
        raise ValueError(
            "Checkpoint and test dataset label mappings do not match."
        )

    print("\nTesting model...")

    test_results = testing_pipeline.test_model(
        model=model,
        test_loader=test_loader,
        index_to_label=checkpoint_index_to_label,
    )

    print("\n")
    print("#" * 70)
    print("TEST RESULTS")
    print("#" * 70)

    print(
        f"Test loss: {test_results['test_loss']:.6f}"
    )
    print(
        f"Accuracy: {test_results['accuracy'] * 100:.2f}%"
    )
    print(
        f"Macro precision: {test_results['macro_precision']:.4f}"
    )
    print(
        f"Macro recall: {test_results['macro_recall']:.4f}"
    )
    print(
        f"Macro F1: {test_results['macro_f1']:.4f}"
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
        f"Weighted F1: {test_results['weighted_f1']:.4f}"
    )

    print("\nClassification Report")
    print("=" * 70)
    print(test_results["classification_report"])

    print("\nConfusion Matrices")
    print("=" * 70)

    for class_index, matrix in enumerate(
        test_results["confusion_matrices"]
    ):
        class_name = checkpoint_index_to_label[class_index]

        print(f"\n{class_name}")
        print(matrix)
