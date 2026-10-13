import copy
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing_pipeline import (
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram
)

from construct_multi_label_dataset import (
    create_train_val_datasets as create_multi_label_train_val_datasets,
    MULTI_LABEL_MAPPING,
    INDEX_TO_MULTI_LABEL,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION
)

from multi_label_transformer_class import (
    MultiLabelTransformer
)


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------

BATCH_SIZE = 16

EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIMS = (256, 512, 256)
NUM_ENCODER_LAYERS = 2

MASK_RATIO = 0.3
ACTIVATION = "relu"

CLASSIFICATION_THRESHOLD = 0.5

SUPERVISED_EPOCHS = 50
SUPERVISED_LEARNING_RATE = 1e-5

PATIENCE = 10

CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIRECTORY = "multi_label_checkpoints"

# Existing unsupervised checkpoint to load.
UNSUPERVISED_CHECKPOINT_PATH = Path(
    "checkpoints/mel_pretrained_fine_tuned_test_unsupervised_best.pt"
)

SET_LIMIT = False
MAXIMUM_TRACKS = 20

DATA_REPRESENTATION = (
    convert_to_mel_spectrogram
)

FINE_TUNED_MODEL_NAME = (
    "mel_multi_label_pretrained_fine_tuned"
)


# ---------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------

class TrainMultiLabelModelPipeline:
    """
    Loads an existing unsupervised Transformer checkpoint
    and continues training with supervised multi-label
    fine-tuning.

    The unsupervised stage is not trained again in this
    script.

    Only supervised loss and element-wise multi-label
    accuracy are tracked.
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dims=HIDDEN_DIMS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_classes=13,
        mask_ratio=MASK_RATIO,
        activation=ACTIVATION,
        classification_threshold=(
            CLASSIFICATION_THRESHOLD
        ),
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_directory=(
            CHECKPOINT_DIRECTORY
        ),
        device=None
    ):
        if checkpoint_interval < 1:
            raise ValueError(
                "checkpoint_interval must be at least 1."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dims = tuple(hidden_dims)
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.activation = activation
        self.classification_threshold = (
            classification_threshold
        )

        self.checkpoint_interval = (
            checkpoint_interval
        )

        self.checkpoint_directory = Path(
            checkpoint_directory
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )


    # -----------------------------------------------------------------
    # Model state helpers
    # -----------------------------------------------------------------

    def _copy_value(
        self,
        value
    ):
        if isinstance(
            value,
            torch.Tensor
        ):

            return (
                value
                .detach()
                .cpu()
                .clone()
            )

        if isinstance(
            value,
            list
        ):

            return [
                self._copy_value(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            tuple
        ):

            return tuple(
                self._copy_value(
                    item
                )
                for item in value
            )

        if isinstance(
            value,
            dict
        ):

            return {
                key:
                    self._copy_value(
                        item
                    )
                for key, item
                in value.items()
            }

        return copy.deepcopy(
            value
        )


    def copy_model_state(
        self,
        model
    ):
        return self._copy_value(
            model.state_dict()
        )


    def _extract_pretrained_state(
        self,
        checkpoint
    ):
        """
        Extracts the best available unsupervised model
        state from several common checkpoint layouts.
        """

        if not isinstance(
            checkpoint,
            dict
        ):

            raise TypeError(
                "Unsupervised checkpoint must be a "
                "dictionary or model state dictionary."
            )

        # Full training checkpoint: prefer the stored
        # best model from unsupervised pretraining.
        if (
            "best_model_state_dict"
            in checkpoint
            and checkpoint[
                "best_model_state_dict"
            ]
            is not None
        ):

            return (
                checkpoint[
                    "best_model_state_dict"
                ]
            )

        # Standard checkpoint current model state.
        if (
            "model_state_dict"
            in checkpoint
            and checkpoint[
                "model_state_dict"
            ]
            is not None
        ):

            return (
                checkpoint[
                    "model_state_dict"
                ]
            )

        # Plain state dictionary.
        if (
            "W_embedding"
            in checkpoint
            and "encoder_layers"
            in checkpoint
        ):

            return checkpoint

        raise ValueError(
            "Could not find a model state inside "
            "the unsupervised checkpoint."
        )


    def _checkpoint_configuration(
        self,
        checkpoint
    ):
        """
        Uses architecture information from the checkpoint
        when it exists. Otherwise the current script
        configuration is used.
        """

        configuration = checkpoint.get(
            "model_configuration",
            {}
        )

        return {
            "embedding_dim":
                configuration.get(
                    "embedding_dim",
                    self.embedding_dim
                ),

            "num_heads":
                configuration.get(
                    "num_heads",
                    self.num_heads
                ),

            "hidden_dims":
                tuple(
                    configuration.get(
                        "hidden_dims",
                        self.hidden_dims
                    )
                ),

            "num_encoder_layers":
                configuration.get(
                    "num_encoder_layers",
                    self.num_encoder_layers
                ),

            "mask_ratio":
                configuration.get(
                    "mask_ratio",
                    self.mask_ratio
                ),

            "activation":
                configuration.get(
                    "activation",
                    self.activation
                )
        }


    def load_unsupervised_checkpoint(
        self,
        checkpoint_path,
        sample_batch
    ):
        """
        Loads the pretrained embedding and Transformer
        encoder from an existing unsupervised .pt file.

        The multi-label classification head remains
        freshly initialized.
        """

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "Unsupervised checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        pretrained_state = (
            self._extract_pretrained_state(
                checkpoint
            )
        )

        configuration = (
            self._checkpoint_configuration(
                checkpoint
            )
        )

        sample_batch = sample_batch.to(
            device=self.device,
            dtype=torch.float32
        )

        checkpoint_patch_dim = (
            checkpoint
            .get(
                "model_configuration",
                {}
            )
            .get(
                "patch_dim"
            )
        )

        if (
            checkpoint_patch_dim
            is not None
            and sample_batch.shape[-1]
            != checkpoint_patch_dim
        ):

            raise ValueError(
                "The supervised dataset patch dimension "
                "does not match the unsupervised checkpoint. "
                f"Dataset: {sample_batch.shape[-1]}, "
                f"checkpoint: {checkpoint_patch_dim}."
            )

        model = MultiLabelTransformer(
            sample_batch=sample_batch,
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
            hidden_dims=(
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
                self.num_classes
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
                self.classification_threshold
            )
        )

        # This deliberately loads the pretrained
        # representation while leaving the new
        # multi-label classifier freshly initialized.
        model.load_pretrained_state_dict(
            pretrained_state,
            load_reconstruction_head=True
        )

        print(
            "\nLoaded unsupervised checkpoint:",
            checkpoint_path
        )

        if isinstance(
            checkpoint,
            dict
        ):

            print(
                "Checkpoint phase:",
                checkpoint.get(
                    "phase",
                    "not stored"
                )
            )

            print(
                "Checkpoint epoch:",
                checkpoint.get(
                    "epoch",
                    "not stored"
                )
            )

            print(
                "Best unsupervised epoch:",
                checkpoint.get(
                    "best_epoch",
                    "not stored"
                )
            )

            print(
                "Best unsupervised validation loss:",
                checkpoint.get(
                    "best_validation_loss",
                    "not stored"
                )
            )

        print(
            "Pretrained patch embedding and encoder "
            "loaded successfully."
        )

        print(
            "Multi-label classification head is "
            "freshly initialized."
        )

        return (
            model,
            checkpoint
        )


    # -----------------------------------------------------------------
    # Optimizer and checkpointing
    # -----------------------------------------------------------------

    def _create_supervised_optimizer(
        self,
        model,
        learning_rate
    ):
        return torch.optim.Adam(
            model.supervised_parameters(),
            lr=learning_rate
        )


    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        model_name,
        label_to_index,
        index_to_label,
        best_validation_loss,
        best_validation_accuracy,
        best_epoch,
        best_model_state,
        source_unsupervised_checkpoint,
        checkpoint_type="latest"
    ):
        if checkpoint_type not in (
            "latest",
            "best"
        ):

            raise ValueError(
                "checkpoint_type must be "
                "'latest' or 'best'."
            )

        checkpoint = {
            "model_name":
                model_name,

            "checkpoint_type":
                checkpoint_type,

            "phase":
                "supervised",

            "epoch":
                epoch,

            "model_state_dict":
                self.copy_model_state(
                    model
                ),

            "best_validation_loss":
                best_validation_loss,

            "best_validation_accuracy":
                best_validation_accuracy,

            "best_epoch":
                best_epoch,

            "best_model_state_dict":
                self._copy_value(
                    best_model_state
                ),

            "source_unsupervised_checkpoint":
                str(
                    source_unsupervised_checkpoint
                ),

            "model_configuration": {
                "patch_dim":
                    model.patch_dim,

                "embedding_dim":
                    model.embedding_dim,

                "num_heads":
                    model.num_heads,

                "hidden_dims":
                    tuple(
                        model.hidden_dims
                    ),

                "num_encoder_layers":
                    model.num_encoder_layers,

                "num_classes":
                    model.num_classes,

                "mask_ratio":
                    model.mask_ratio,

                "activation":
                    model.activation,

                "classification_threshold":
                    model.classification_threshold
            },

            "label_to_index":
                dict(
                    label_to_index
                ),

            "index_to_label":
                dict(
                    index_to_label
                ),

            "saved_at":
                datetime.now().isoformat(
                    timespec="seconds"
                )
        }

        if checkpoint_type == "latest":

            checkpoint[
                "optimizer_state_dict"
            ] = (
                self._copy_value(
                    optimizer.state_dict()
                )
            )

            file_name = (
                f"{model_name}_"
                "supervised_latest.pt"
            )

        else:

            file_name = (
                f"{model_name}_"
                "supervised_best.pt"
            )

        file_path = (
            self.checkpoint_directory
            / file_name
        )

        temporary_path = (
            file_path.with_suffix(
                ".tmp"
            )
        )

        torch.save(
            checkpoint,
            temporary_path
        )

        temporary_path.replace(
            file_path
        )

        return file_path


    # -----------------------------------------------------------------
    # Multi-label accuracy
    # -----------------------------------------------------------------

    def _calculate_accuracy(
        self,
        predictions,
        labels
    ):
        """
        Element-wise multi-label accuracy.

        Every class decision counts as one binary
        prediction.
        """

        predictions = (
            predictions
            .to(
                dtype=torch.bool
            )
        )

        labels = (
            labels
            .to(
                dtype=torch.bool
            )
        )

        correct = (
            predictions
            == labels
        ).sum().item()

        total = (
            labels.numel()
        )

        return (
            correct,
            total
        )


    # -----------------------------------------------------------------
    # Supervised training
    # -----------------------------------------------------------------

    def train_supervised_epoch(
        self,
        model,
        data_loader,
        optimizer
    ):
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
        number_of_batches = 0

        for (
            patches,
            labels
        ) in data_loader:

            patches = patches.to(
                device=self.device,
                dtype=torch.float32
            )

            labels = labels.to(
                device=self.device,
                dtype=torch.float32
            )

            optimizer.zero_grad()

            (
                loss,
                logits,
                predicted_labels,
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
                    "Supervised training loss "
                    "became NaN or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            (
                correct,
                total
            ) = (
                self._calculate_accuracy(
                    predicted_labels,
                    labels
                )
            )

            total_correct += correct
            total_labels += total
            number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Supervised training DataLoader "
                "contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        accuracy = (
            total_correct
            / total_labels
        )

        return (
            average_loss,
            accuracy
        )


    def validate_supervised_epoch(
        self,
        model,
        data_loader
    ):
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
        number_of_batches = 0

        with torch.no_grad():

            for (
                patches,
                labels
            ) in data_loader:

                patches = patches.to(
                    device=self.device,
                    dtype=torch.float32
                )

                labels = labels.to(
                    device=self.device,
                    dtype=torch.float32
                )

                (
                    loss,
                    logits,
                    predicted_labels,
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
                        "Supervised validation "
                        "loss became NaN or infinite."
                    )

                total_loss += (
                    loss.item()
                )

                (
                    correct,
                    total
                ) = (
                    self._calculate_accuracy(
                        predicted_labels,
                        labels
                    )
                )

                total_correct += correct
                total_labels += total
                number_of_batches += 1

        if number_of_batches == 0:

            raise ValueError(
                "Supervised validation DataLoader "
                "contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        accuracy = (
            total_correct
            / total_labels
        )

        return (
            average_loss,
            accuracy
        )


    def fine_tune_supervised(
        self,
        model,
        training_loader,
        validation_loader,
        epochs,
        learning_rate,
        model_name,
        label_to_index,
        index_to_label,
        source_unsupervised_checkpoint,
        patience=PATIENCE
    ):
        """
        Fine-tunes the pretrained Transformer on the
        multi-label mixture dataset.
        """

        optimizer = (
            self._create_supervised_optimizer(
                model=model,
                learning_rate=(
                    learning_rate
                )
            )
        )

        history = []
        latest_checkpoint_paths = []

        best_validation_loss = float(
            "inf"
        )

        best_validation_accuracy = 0.0
        best_epoch = None

        best_model_state = (
            self.copy_model_state(
                model
            )
        )

        patience_count = 0

        for epoch in tqdm(
            range(
                1,
                epochs + 1
            ),
            desc="Multi-label supervised fine-tuning"
        ):

            (
                training_loss,
                training_accuracy
            ) = (
                self.train_supervised_epoch(
                    model=model,
                    data_loader=(
                        training_loader
                    ),
                    optimizer=optimizer
                )
            )

            (
                validation_loss,
                validation_accuracy
            ) = (
                self.validate_supervised_epoch(
                    model=model,
                    data_loader=(
                        validation_loader
                    )
                )
            )

            if (
                validation_loss
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
                )

                best_validation_accuracy = (
                    validation_accuracy
                )

                best_epoch = epoch

                best_model_state = (
                    self.copy_model_state(
                        model
                    )
                )

                patience_count = 0

            else:

                patience_count += 1

            history.append({
                "epoch":
                    epoch,

                "training_loss":
                    training_loss,

                "training_accuracy":
                    training_accuracy,

                "validation_loss":
                    validation_loss,

                "validation_accuracy":
                    validation_accuracy,

                "best_validation_loss":
                    best_validation_loss,

                "best_validation_accuracy":
                    best_validation_accuracy,

                "best_epoch":
                    best_epoch
            })

            print(
                f"Epoch {epoch} | "
                f"Train loss: "
                f"{training_loss:.6f} | "
                f"Train accuracy: "
                f"{training_accuracy * 100:.2f}% | "
                f"Val loss: "
                f"{validation_loss:.6f} | "
                f"Val accuracy: "
                f"{validation_accuracy * 100:.2f}%"
            )

            if (
                epoch
                % self.checkpoint_interval
                == 0
            ):

                checkpoint_path = (
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        model_name=model_name,
                        label_to_index=(
                            label_to_index
                        ),
                        index_to_label=(
                            index_to_label
                        ),
                        best_validation_loss=(
                            best_validation_loss
                        ),
                        best_validation_accuracy=(
                            best_validation_accuracy
                        ),
                        best_epoch=(
                            best_epoch
                        ),
                        best_model_state=(
                            best_model_state
                        ),
                        source_unsupervised_checkpoint=(
                            source_unsupervised_checkpoint
                        ),
                        checkpoint_type="latest"
                    )
                )

                latest_checkpoint_paths.append(
                    checkpoint_path
                )

            if (
                patience_count
                >= patience
            ):

                print(
                    "Validation loss early stop "
                    "activated."
                )

                break

        model.load_state_dict(
            best_model_state
        )

        best_checkpoint_path = (
            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                model_name=model_name,
                label_to_index=(
                    label_to_index
                ),
                index_to_label=(
                    index_to_label
                ),
                best_validation_loss=(
                    best_validation_loss
                ),
                best_validation_accuracy=(
                    best_validation_accuracy
                ),
                best_epoch=(
                    best_epoch
                ),
                best_model_state=(
                    best_model_state
                ),
                source_unsupervised_checkpoint=(
                    source_unsupervised_checkpoint
                ),
                checkpoint_type="best"
            )
        )

        return {
            "model":
                model,

            "optimizer":
                optimizer,

            "history":
                history,

            "best_validation_loss":
                best_validation_loss,

            "best_validation_accuracy":
                best_validation_accuracy,

            "best_epoch":
                best_epoch,

            "best_model_state":
                best_model_state,

            "latest_checkpoint_paths":
                latest_checkpoint_paths,

            "best_checkpoint_path":
                best_checkpoint_path
        }


# ---------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nCreating multi-label mixture datasets..."
    )

    (
        multi_label_training_dataset,
        multi_label_validation_dataset
    ) = (
        create_multi_label_train_val_datasets(
            training_path=(
                SLAKH2100_REDUX_16K_TRAIN
            ),
            validation_path=(
                SLAKH2100_REDUX_16K_VALIDATION
            ),
            data_representation=(
                DATA_REPRESENTATION
            ),
            set_limit=(
                SET_LIMIT
            ),
            maximum_tracks=(
                MAXIMUM_TRACKS
            ),
            show_progress=True
        )
    )

    (
        label_to_index,
        index_to_label
    ) = (
        multi_label_training_dataset
        .return_label_mappings()
    )

    if (
        label_to_index
        != MULTI_LABEL_MAPPING
    ):

        raise ValueError(
            "Multi-label training dataset mapping "
            "does not match MULTI_LABEL_MAPPING."
        )

    if (
        index_to_label
        != INDEX_TO_MULTI_LABEL
    ):

        raise ValueError(
            "Multi-label index mapping does not "
            "match INDEX_TO_MULTI_LABEL."
        )

    num_classes = len(
        label_to_index
    )

    supervised_training_loader = (
        DataLoader(
            multi_label_training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
    )

    supervised_validation_loader = (
        DataLoader(
            multi_label_validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    (
        supervised_sample_batch,
        supervised_label_batch
    ) = next(
        iter(
            supervised_training_loader
        )
    )

    print(
        "\nSupervised mixture batch shape:",
        supervised_sample_batch.shape
    )

    print(
        "Supervised label batch shape:",
        supervised_label_batch.shape
    )

    training_pipeline = (
        TrainMultiLabelModelPipeline(
            embedding_dim=(
                EMBEDDING_DIM
            ),
            num_heads=(
                NUM_HEADS
            ),
            hidden_dims=(
                HIDDEN_DIMS
            ),
            num_encoder_layers=(
                NUM_ENCODER_LAYERS
            ),
            num_classes=(
                num_classes
            ),
            mask_ratio=(
                MASK_RATIO
            ),
            activation=(
                ACTIVATION
            ),
            classification_threshold=(
                CLASSIFICATION_THRESHOLD
            ),
            checkpoint_interval=(
                CHECKPOINT_INTERVAL
            ),
            checkpoint_directory=(
                CHECKPOINT_DIRECTORY
            )
        )
    )

    print(
        "\nTraining device:",
        training_pipeline.device
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "\n"
        + "#" * 70
    )

    print(
        "LOADING EXISTING UNSUPERVISED MODEL"
    )

    print(
        "#" * 70
    )

    (
        model,
        unsupervised_checkpoint
    ) = (
        training_pipeline
        .load_unsupervised_checkpoint(
            checkpoint_path=(
                UNSUPERVISED_CHECKPOINT_PATH
            ),
            sample_batch=(
                supervised_sample_batch
            )
        )
    )

    print(
        "\n"
        + "#" * 70
    )

    print(
        "MULTI-LABEL SUPERVISED FINE-TUNING"
    )

    print(
        "#" * 70
    )

    fine_tuning_results = (
        training_pipeline
        .fine_tune_supervised(
            model=model,
            training_loader=(
                supervised_training_loader
            ),
            validation_loader=(
                supervised_validation_loader
            ),
            epochs=(
                SUPERVISED_EPOCHS
            ),
            learning_rate=(
                SUPERVISED_LEARNING_RATE
            ),
            model_name=(
                FINE_TUNED_MODEL_NAME
            ),
            label_to_index=(
                label_to_index
            ),
            index_to_label=(
                index_to_label
            ),
            source_unsupervised_checkpoint=(
                UNSUPERVISED_CHECKPOINT_PATH
            )
        )
    )

    print(
        "\nBest multi-label validation loss:",
        fine_tuning_results[
            "best_validation_loss"
        ]
    )

    print(
        "Best multi-label validation accuracy:",
        (
            fine_tuning_results[
                "best_validation_accuracy"
            ]
            * 100
        )
    )

    print(
        "Best fine-tuning epoch:",
        fine_tuning_results[
            "best_epoch"
        ]
    )

    print(
        "Best checkpoint:",
        fine_tuning_results[
            "best_checkpoint_path"
        ]
    )

    print(
        "\nTraining complete."
    )