import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from transformer_model_class import Transformer
from unsupervised_dataset_class import (
    create_unsupervised_datasets,
    UNSUPERVISED_DATASET_PATHS
)
from supervised_dataset_class import (
    create_supervised_datasets,
    BABY_SLAKH_DATASET_PATH
)

# Train Config
UNSUPERVISED_EPOCHS = 5
SUPERVISED_EPOCHS = 5
UNSUPERVISED_LEARNING_RATE = 1e-3
SUPERVISED_LEARNING_RATE = 1e-3
BATCH_SIZE = 16

# Model Config
PATCH_DIM = 1280
EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 512
MASK_RATIO = 0.30

# Output folders
MODEL_OUTPUT_DIRECTORY = Path("saved_models")
RESULT_OUTPUT_DIRECTORY = Path("test_results")

MODEL_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

PRETRAINED_MODEL_PATH = (
    MODEL_OUTPUT_DIRECTORY
    / "best_pretrained_fine_tuned_model.pt"
)

SUPERVISED_ONLY_MODEL_PATH = (
    MODEL_OUTPUT_DIRECTORY
    / "best_supervised_only_model.pt"
)

PRETRAINED_RESULTS_PDF_PATH = (
    RESULT_OUTPUT_DIRECTORY
    / "pretrained_fine_tuned_test_results.pdf"
)

SUPERVISED_ONLY_RESULTS_PDF_PATH = (
    RESULT_OUTPUT_DIRECTORY
    / "supervised_only_test_results.pdf"
)

# Training and evaluation pipeline
class TrainModelPipeline:
    """
    Trains and evaluates the custom Transformer model.

    Supported training approaches:

    1. Unsupervised pretraining followed by supervised
       fine-tuning.

    2. Supervised-only training with a newly initialized
       Transformer.
    """
    def __init__(
        self,
        patch_dim,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_classes,
        mask_ratio=0.30
    ):
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio

    def create_model(self):
        """
        Creates a newly initialized Transformer.
        """

        model = Transformer(
            patch_dim=self.patch_dim,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            mask_ratio=self.mask_ratio
        )

        return model

    def copy_model_state(self, model):
        """
        Creates a detached copy of the model parameters.
        """

        copied_state = {}

        for parameter_name, parameter_value in (
            model.state_dict().items()
        ):
            copied_state[parameter_name] = (
                parameter_value
                .detach()
                .clone()
            )

        return copied_state

    def save_model_checkpoint(
        self,
        model,
        file_path,
        model_name,
        label_to_index,
        index_to_label,
        validation_loss=None,
        validation_accuracy=None
    ):
        """
        Saves the custom Transformer parameters and
        supporting configuration to a .pt checkpoint.
        """

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        model_state = self.copy_model_state(
            model
        )

        checkpoint = {
            "model_name": model_name,
            "model_state_dict": model_state,

            "model_configuration": {
                "patch_dim": self.patch_dim,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
                "mask_ratio": self.mask_ratio
            },

            "label_to_index": dict(
                label_to_index
            ),

            "index_to_label": dict(
                index_to_label
            ),

            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,

            "saved_at": datetime.now().isoformat(
                timespec="seconds"
            )
        }

        torch.save(
            checkpoint,
            file_path
        )

        print(
            f"Saved model checkpoint: "
            f"{file_path}"
        )

    def train_unsupervised_epoch(
        self,
        model,
        data_loader,
        optimizer
    ):
        """
        Runs one epoch of masked-patch reconstruction
        training.
        """

        total_loss = 0.0
        number_of_batches = 0

        for patches in data_loader:
            patches = patches.to(
                dtype=torch.float32
            )

            optimizer.zero_grad()

            (
                loss,
                reconstructed_patches,
                mask,
                attention_weights
            ) = model.unsupervised_reconstruction(
                patches
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "The unsupervised loss became NaN "
                    "or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            number_of_batches += 1

        if number_of_batches == 0:
            return 0.0

        average_loss = (
            total_loss / number_of_batches
        )

        return average_loss

    def validate_unsupervised_epoch(
        self,
        model,
        data_loader
    ):
        """
        Runs one epoch of masked-patch reconstruction
        validation.
        """

        total_loss = 0.0
        number_of_batches = 0

        with torch.no_grad():
            for patches in data_loader:
                patches = patches.to(
                    dtype=torch.float32
                )

                (
                    loss,
                    reconstructed_patches,
                    mask,
                    attention_weights
                ) = model.unsupervised_reconstruction(
                    patches
                )

                total_loss += loss.item()
                number_of_batches += 1

        if number_of_batches == 0:
            return 0.0

        average_loss = (
            total_loss / number_of_batches
        )

        return average_loss

    def pretrain_unsupervised(
        self,
        model,
        training_loader,
        validation_loader,
        epochs,
        learning_rate
    ):
        """
        Performs masked-patch reconstruction pretraining.
        """

        optimizer = torch.optim.Adam(
            model.unsupervised_parameters(),
            lr=learning_rate
        )

        best_validation_loss = float("inf")
        best_model_state = None

        print("\nUnsupervised pretraining")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            training_loss = (
                self.train_unsupervised_epoch(
                    model,
                    training_loader,
                    optimizer
                )
            )

            validation_loss = (
                self.validate_unsupervised_epoch(
                    model,
                    validation_loader
                )
            )

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Training loss: "
                f"{training_loss:.6f} | "
                f"Validation loss: "
                f"{validation_loss:.6f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                best_model_state = (
                    self.copy_model_state(
                        model
                    )
                )

                print(
                    "Saved best unsupervised "
                    "model parameters."
                )

        if best_model_state is not None:
            model.load_state_dict(
                best_model_state
            )

        print(
            "\nBest unsupervised validation loss: "
            f"{best_validation_loss:.6f}"
        )

        return model

    def train_supervised_epoch(
        self,
        model,
        data_loader,
        optimizer
    ):
        """
        Runs one epoch of supervised classification
        training.
        """

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        number_of_batches = 0

        for patches, labels in data_loader:
            patches = patches.to(
                dtype=torch.float32
            )

            labels = labels.to(
                dtype=torch.long
            )

            optimizer.zero_grad()

            (
                loss,
                logits,
                predicted_classes,
                probabilities,
                attention_weights
            ) = model.supervised_fine_tuning(
                patches,
                labels
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "The supervised loss became NaN "
                    "or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total_correct += (
                predicted_classes == labels
            ).sum().item()

            total_samples += labels.shape[0]
            number_of_batches += 1

        if number_of_batches == 0:
            return 0.0, 0.0

        average_loss = (
            total_loss / number_of_batches
        )

        accuracy = (
            total_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        return average_loss, accuracy

    def validate_supervised_epoch(
        self,
        model,
        data_loader
    ):
        """
        Runs one epoch of supervised classification
        validation.
        """

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        number_of_batches = 0

        with torch.no_grad():
            for patches, labels in data_loader:
                patches = patches.to(
                    dtype=torch.float32
                )

                labels = labels.to(
                    dtype=torch.long
                )

                (
                    loss,
                    logits,
                    predicted_classes,
                    probabilities,
                    attention_weights
                ) = model.supervised_fine_tuning(
                    patches,
                    labels
                )

                total_loss += loss.item()

                total_correct += (
                    predicted_classes == labels
                ).sum().item()

                total_samples += labels.shape[0]
                number_of_batches += 1

        if number_of_batches == 0:
            return 0.0, 0.0

        average_loss = (
            total_loss / number_of_batches
        )

        accuracy = (
            total_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        return average_loss, accuracy

    def fine_tune_supervised(
        self,
        model,
        training_loader,
        validation_loader,
        epochs,
        learning_rate,
        model_path,
        model_name,
        label_to_index,
        index_to_label,
    ):
        """
        Trains a Transformer on labelled data.
        """

        optimizer = torch.optim.Adam(
            model.supervised_parameters(),
            lr=learning_rate
        )

        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0
        best_model_state = None

        print("\nSupervised training")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            (
                training_loss,
                training_accuracy
            ) = self.train_supervised_epoch(
                model,
                training_loader,
                optimizer
            )

            (
                validation_loss,
                validation_accuracy
            ) = self.validate_supervised_epoch(
                model,
                validation_loader
            )

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Training loss: "
                f"{training_loss:.6f} | "
                f"Training accuracy: "
                f"{training_accuracy * 100:.2f}% | "
                f"Validation loss: "
                f"{validation_loss:.6f} | "
                f"Validation accuracy: "
                f"{validation_accuracy * 100:.2f}%"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = (
                    validation_loss
                )

                best_validation_accuracy = (
                    validation_accuracy
                )

                best_model_state = (
                    self.copy_model_state(
                        model
                    )
                )

                self.save_model_checkpoint(
                    model=model,
                    file_path=model_path,
                    model_name=model_name,
                    label_to_index=label_to_index,
                    index_to_label=index_to_label,
                    validation_loss=(
                        best_validation_loss
                    ),
                    validation_accuracy=(
                        best_validation_accuracy
                    )
                )

                print(
                    "Saved best supervised "
                    "model parameters."
                )

        if best_model_state is not None:
            model.load_state_dict(
                best_model_state
            )

        print(
            "\nBest supervised validation loss: "
            f"{best_validation_loss:.6f}"
        )

        print(
            "Best supervised validation accuracy: "
            f"{best_validation_accuracy * 100:.2f}%"
        )

        return model

    def train_pretrained_and_fine_tuned_model(
        self,
        unsupervised_training_loader,
        unsupervised_validation_loader,
        supervised_training_loader,
        supervised_validation_loader,
        unsupervised_epochs,
        supervised_epochs,
        unsupervised_learning_rate,
        supervised_learning_rate,
        model_path,
        label_to_index,
        index_to_label
    ):
        """
        Creates a Transformer, performs unsupervised
        pretraining, and then performs supervised
        fine-tuning.
        """

        model = self.create_model()

        model = self.pretrain_unsupervised(
            model=model,
            training_loader=(
                unsupervised_training_loader
            ),
            validation_loader=(
                unsupervised_validation_loader
            ),
            epochs=unsupervised_epochs,
            learning_rate=(
                unsupervised_learning_rate
            )
        )

        model = self.fine_tune_supervised(
            model=model,
            training_loader=(
                supervised_training_loader
            ),
            validation_loader=(
                supervised_validation_loader
            ),
            epochs=supervised_epochs,
            learning_rate=(
                supervised_learning_rate
            ),
            model_path=model_path,
            model_name=(
                "Pretrained and fine-tuned Transformer"
            ),
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )

        return model

    def train_supervised_only_model(
        self,
        supervised_training_loader,
        supervised_validation_loader,
        epochs,
        learning_rate,
        model_path,
        label_to_index,
        index_to_label
    ):
        """
        Creates a newly initialized Transformer and trains
        it only on the supervised dataset.
        """

        model = self.create_model()

        model = self.fine_tune_supervised(
            model=model,
            training_loader=(
                supervised_training_loader
            ),
            validation_loader=(
                supervised_validation_loader
            ),
            epochs=epochs,
            learning_rate=learning_rate,
            model_path=model_path,
            model_name=(
                "Supervised-only Transformer"
            ),
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )

        return model

    def test_model(
        self,
        model,
        test_loader,
        model_name,
        index_to_label
    ):
        """
        Evaluates a trained Transformer on the test dataset
        using scikit-learn classification metrics.
        """

        total_loss = 0.0
        number_of_batches = 0

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for patches, labels in test_loader:
                patches = patches.to(
                    dtype=torch.float32
                )

                labels = labels.to(
                    dtype=torch.long
                )

                (
                    loss,
                    logits,
                    predicted_classes,
                    probabilities,
                    attention_weights
                ) = model.supervised_fine_tuning(
                    patches,
                    labels
                )

                if not torch.isfinite(loss):
                    raise ValueError(
                        "The test loss became NaN or infinite."
                    )

                total_loss += loss.item()
                number_of_batches += 1

                true_labels.extend(
                    labels
                    .detach()
                    .cpu()
                    .tolist()
                )

                predicted_labels.extend(
                    predicted_classes
                    .detach()
                    .cpu()
                    .tolist()
                )

        if number_of_batches == 0:
            raise ValueError(
                "The test DataLoader contains no batches."
            )

        if len(true_labels) == 0:
            raise ValueError(
                "The test dataset contains no labelled samples."
            )

        average_test_loss = (
            total_loss / number_of_batches
        )

        class_indices = list(
            range(self.num_classes)
        )

        class_names = [
            index_to_label.get(
                class_index,
                str(class_index)
            )
            for class_index in class_indices
        ]

        test_accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        (
            macro_precision,
            macro_recall,
            macro_f1,
            _
        ) = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=class_indices,
            average="macro",
            zero_division=0
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _
        ) = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=class_indices,
            average="weighted",
            zero_division=0
        )

        report_text = classification_report(
            true_labels,
            predicted_labels,
            labels=class_indices,
            target_names=class_names,
            digits=4,
            zero_division=0
        )

        report_dictionary = classification_report(
            true_labels,
            predicted_labels,
            labels=class_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        confusion_matrix_values = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=class_indices
        )

        print(f"\nTest results: {model_name}")
        print("=" * 80)

        print(
            f"Test loss: "
            f"{average_test_loss:.6f}"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print("\nMacro-average metrics")
        print("-" * 80)

        print(
            f"Precision: "
            f"{macro_precision:.4f}"
        )

        print(
            f"Recall: "
            f"{macro_recall:.4f}"
        )

        print(
            f"F1-score: "
            f"{macro_f1:.4f}"
        )

        print("\nWeighted-average metrics")
        print("-" * 80)

        print(
            f"Precision: "
            f"{weighted_precision:.4f}"
        )

        print(
            f"Recall: "
            f"{weighted_recall:.4f}"
        )

        print(
            f"F1-score: "
            f"{weighted_f1:.4f}"
        )

        print("\nClassification report")
        print("-" * 80)
        print(report_text)

        print("Confusion matrix")
        print("-" * 80)
        print(confusion_matrix_values)

        return {
            "model_name": model_name,
            "test_loss": average_test_loss,
            "accuracy": test_accuracy,

            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,

            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,

            "classification_report": report_dictionary,
            "confusion_matrix": confusion_matrix_values,

            "true_labels": true_labels,
            "predicted_labels": predicted_labels
        }

    def save_test_results_pdf(
        self,
        test_results,
        file_path,
        index_to_label
    ):
        """
        Saves model test results, classification metrics,
        and the confusion matrix to a PDF file.
        """

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        document = SimpleDocTemplate(
            str(file_path),
            pagesize=landscape(A4),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )

        styles = getSampleStyleSheet()
        elements = []

        model_name = test_results[
            "model_name"
        ]

        elements.append(
            Paragraph(
                f"Test Results: {model_name}",
                styles["Title"]
            )
        )

        elements.append(
            Spacer(1, 12)
        )

        elements.append(
            Paragraph(
                "Overall Metrics",
                styles["Heading2"]
            )
        )

        overall_metrics_data = [
            [
                "Metric",
                "Value"
            ],
            [
                "Test loss",
                f"{test_results['test_loss']:.6f}"
            ],
            [
                "Accuracy",
                f"{test_results['accuracy'] * 100:.2f}%"
            ],
            [
                "Macro precision",
                f"{test_results['macro_precision']:.4f}"
            ],
            [
                "Macro recall",
                f"{test_results['macro_recall']:.4f}"
            ],
            [
                "Macro F1-score",
                f"{test_results['macro_f1']:.4f}"
            ],
            [
                "Weighted precision",
                f"{test_results['weighted_precision']:.4f}"
            ],
            [
                "Weighted recall",
                f"{test_results['weighted_recall']:.4f}"
            ],
            [
                "Weighted F1-score",
                f"{test_results['weighted_f1']:.4f}"
            ]
        ]

        overall_metrics_table = Table(
            overall_metrics_data,
            colWidths=[180, 180]
        )

        overall_metrics_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, 0),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, 0),
                    8
                )
            ])
        )

        elements.append(
            overall_metrics_table
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Classification Report",
                styles["Heading2"]
            )
        )

        report = test_results[
            "classification_report"
        ]

        report_table_data = [
            [
                "Class",
                "Precision",
                "Recall",
                "F1-score",
                "Support"
            ]
        ]

        class_names = [
            index_to_label[index]
            for index in range(
                self.num_classes
            )
        ]

        for class_name in class_names:
            class_results = report.get(
                class_name,
                {}
            )

            report_table_data.append([
                class_name,
                (
                    f"{class_results.get('precision', 0.0):.4f}"
                ),
                (
                    f"{class_results.get('recall', 0.0):.4f}"
                ),
                (
                    f"{class_results.get('f1-score', 0.0):.4f}"
                ),
                int(
                    class_results.get(
                        "support",
                        0
                    )
                )
            ])

        for average_name, display_name in [
            ("macro avg", "Macro average"),
            ("weighted avg", "Weighted average")
        ]:
            average_results = report.get(
                average_name,
                {}
            )

            report_table_data.append([
                display_name,
                (
                    f"{average_results.get('precision', 0.0):.4f}"
                ),
                (
                    f"{average_results.get('recall', 0.0):.4f}"
                ),
                (
                    f"{average_results.get('f1-score', 0.0):.4f}"
                ),
                int(
                    average_results.get(
                        "support",
                        0
                    )
                )
            ])

        report_table = Table(
            report_table_data,
            repeatRows=1,
            colWidths=[
                150,
                90,
                90,
                90,
                80
            ]
        )

        report_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        elements.append(
            report_table
        )

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "Confusion Matrix",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                "Rows represent true classes. "
                "Columns represent predicted classes.",
                styles["BodyText"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        confusion_matrix_values = (
            test_results[
                "confusion_matrix"
            ]
        )

        confusion_header = [
            "True / Pred."
        ] + [
            str(index)
            for index in range(
                self.num_classes
            )
        ]

        confusion_table_data = [
            confusion_header
        ]

        for class_index, matrix_row in enumerate(
            confusion_matrix_values
        ):
            confusion_table_data.append(
                [
                    str(class_index)
                ] + [
                    int(value)
                    for value in matrix_row
                ]
            )

        number_of_columns = (
            self.num_classes + 1
        )

        available_width = (
            landscape(A4)[0] - 60
        )

        column_width = (
            available_width
            / number_of_columns
        )

        confusion_table = Table(
            confusion_table_data,
            repeatRows=1,
            colWidths=[
                column_width
            ] * number_of_columns
        )

        confusion_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.black
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        elements.append(
            confusion_table
        )

        elements.append(
            Spacer(1, 16)
        )

        elements.append(
            Paragraph(
                "Class Index Key",
                styles["Heading3"]
            )
        )

        class_key_data = [
            ["Index", "Class"]
        ]

        for class_index in range(
            self.num_classes
        ):
            class_key_data.append([
                class_index,
                index_to_label[
                    class_index
                ]
            ])

        class_key_table = Table(
            class_key_data,
            repeatRows=1,
            colWidths=[80, 220]
        )

        class_key_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (0, -1),
                    "CENTER"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        elements.append(
            class_key_table
        )

        document.build(
            elements
        )

        print(
            f"Saved test-results PDF: "
            f"{file_path}"
        )


if __name__ == "__main__":
    # Datasets
    unsupervised_train_dataset, unsupervised_val_dataset= create_unsupervised_datasets(UNSUPERVISED_DATASET_PATHS)
    supervised_training_dataset, supervised_validation_dataset, supervised_test_dataset, label_to_index, index_to_label = create_supervised_datasets(BABY_SLAKH_DATASET_PATH)
    num_classes = len(label_to_index)

    # Check datasets
    print("Dataset Information:...........................................................................................")
    print(f"Unsupervised training dataset size: {len(unsupervised_train_dataset)}")
    print(f"Unsupervised validation dataset size: {len(unsupervised_val_dataset)}")
    print(f"Supervised training dataset size: {len(supervised_training_dataset)}")
    print(f"Supervised validation dataset size: {len(supervised_validation_dataset)}")
    print(f"Supervised test dataset size: {len(supervised_test_dataset)}")
    print(f"Label to index mapping: {label_to_index}")
    print(f"Index to label mapping: {index_to_label}")
    print(f"Number of classes: {num_classes}")
    print("..................................................................................................................")

    # DataLoaders
    unsupervised_train_loader = DataLoader(
        unsupervised_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    unsupervised_val_loader = DataLoader(
        unsupervised_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    supervised_train_loader = DataLoader(
        supervised_training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    supervised_val_loader = DataLoader(
        supervised_validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    supervised_test_loader = DataLoader(
        supervised_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Train and evaluate the model
    training_pipeline = TrainModelPipeline(
        patch_dim=PATCH_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        mask_ratio=MASK_RATIO
    )

    # Train and evaluate the model with unsupervised pretraining followed by supervised fine-tuning
    print("\n")
    print("#" * 70)
    print(
        "MODEL 1: UNSUPERVISED PRETRAINING "
        "FOLLOWED BY SUPERVISED FINE-TUNING"
    )
    print("#" * 70)

    pretrained_fine_tuned_model = (
        training_pipeline
        .train_pretrained_and_fine_tuned_model(
            unsupervised_training_loader=(
                unsupervised_train_loader
            ),
            unsupervised_validation_loader=(
                unsupervised_val_loader
            ),
            supervised_training_loader=(
                supervised_train_loader
            ),
            supervised_validation_loader=(
                supervised_val_loader
            ),
            unsupervised_epochs=(
                UNSUPERVISED_EPOCHS
            ),
            supervised_epochs=(
                SUPERVISED_EPOCHS
            ),
            unsupervised_learning_rate=(
                UNSUPERVISED_LEARNING_RATE
            ),
            supervised_learning_rate=(
                SUPERVISED_LEARNING_RATE
            ),
            model_path=(
                PRETRAINED_MODEL_PATH
            ),
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )
    )

    pretrained_test_results = (
        training_pipeline.test_model(
            model=pretrained_fine_tuned_model,
            test_loader=supervised_test_loader,
            model_name=(
                "Pretrained and fine-tuned Transformer"
            ),
            index_to_label=index_to_label
        )
    )

    training_pipeline.save_test_results_pdf(
        test_results=pretrained_test_results,
        file_path=(
            PRETRAINED_RESULTS_PDF_PATH
        ),
        index_to_label=index_to_label
    )

    # Train and evaluate the model with supervised fine-tuning only
    print("\n")
    print("#" * 70)
    print(
        "MODEL 2: SUPERVISED-ONLY TRAINING"
    )
    print("#" * 70)

    supervised_only_model = (
        training_pipeline
        .train_supervised_only_model(
            supervised_training_loader=(
                supervised_train_loader
            ),
            supervised_validation_loader=(
                supervised_val_loader
            ),
            epochs=SUPERVISED_EPOCHS,
            learning_rate=(
                SUPERVISED_LEARNING_RATE
            ),
            model_path=(
                SUPERVISED_ONLY_MODEL_PATH
            ),
            label_to_index=label_to_index,
            index_to_label=index_to_label
        )
    )

    supervised_only_test_results = (
        training_pipeline.test_model(
            model=supervised_only_model,
            test_loader=supervised_test_loader,
            model_name=(
                "Supervised-only Transformer"
            ),
            index_to_label=index_to_label
        )
    )

    training_pipeline.save_test_results_pdf(
        test_results=(
            supervised_only_test_results
        ),
        file_path=(
            SUPERVISED_ONLY_RESULTS_PDF_PATH
        ),
        index_to_label=index_to_label
    )