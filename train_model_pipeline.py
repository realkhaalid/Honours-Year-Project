import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from transformer_model_class import Transformer
from data_processing_pipeline import (
    convert_to_mel_spectrogram,
    convert_to_cqt_spectrogram,
    convert_to_stft_spectrogram
)
from construct_dataset_class import (
    create_datasets,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION,
    SLAKH2100_REDUX_16K_TEST
)

# Train Config
BATCH_SIZE = 16

EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIMS = (256,512,256)
NUM_ENCODER_LAYERS = 2

MASK_RATIO = 0.3
ACTIVATION = "relu"

UNSUPERVISED_EPOCHS = 30
SUPERVISED_EPOCHS = 50

UNSUPERVISED_LEARNING_RATE = 1e-4
SUPERVISED_LEARNING_RATE = 1e-5
PATIENCE = 10

CHECKPOINT_INTERVAL = 10
CHECKPOINT_DIRECTORY = "checkpoints"

SET_LIMIT = False
MAXIMUM_TRACKS = 500

DATA_REPRESENTATION = (
    convert_to_mel_spectrogram
)

PRETRAINED_MODEL_NAME = (
    "mel_pretrained_fine_tuned_test"
)

SUPERVISED_ONLY_MODEL_NAME = (
    "mel_supervised_only_test"
)

# Training and evaluation pipeline
class TrainModelPipeline:
    """
    Trains the custom Transformer model.

    Supported training approaches:

    1. Unsupervised pretraining followed by supervised
       fine-tuning.

    2. Supervised-only training with a newly initialized
       Transformer.
    """
    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        hidden_dims=(512,),
        num_encoder_layers=2,
        num_classes=13,
        mask_ratio=0.3,
        activation="gelu",
        checkpoint_interval=10,
        checkpoint_directory="checkpoints",
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

    def create_model(
        self,
        sample_batch
    ):
        """
        Creates a new Transformer using the patch
        dimension obtained directly from the data.
        """

        sample_batch = sample_batch.to(
            device=self.device,
            dtype=torch.float32
        )

        model = Transformer(
            sample_batch=sample_batch,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            hidden_dims=self.hidden_dims,
            num_encoder_layers=(
                self.num_encoder_layers
            ),
            num_classes=self.num_classes,
            mask_ratio=self.mask_ratio,
            activation=self.activation
        )

        return model

    def _copy_value(
        self,
        value
    ):
        """
        Recursively creates detached CPU copies
        of model-state values.
        """

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
                self._copy_value(item)
                for item in value
            ]

        if isinstance(
            value,
            tuple
        ):
            return tuple(
                self._copy_value(item)
                for item in value
            )

        if isinstance(
            value,
            dict
        ):
            return {
                key: self._copy_value(item)
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
        """
        Returns an independent CPU copy of the
        current model parameters.
        """

        return self._copy_value(
            model.state_dict()
        )

    def _create_optimizer(
        self,
        model,
        phase,
        learning_rate
    ):
        """
        Creates the correct Adam optimizer for
        unsupervised or supervised training.
        """

        if phase == "unsupervised":

            parameters = (
                model.unsupervised_parameters()
            )

        elif phase == "supervised":

            parameters = (
                model.supervised_parameters()
            )

        else:
            raise ValueError(
                f"Unknown training phase: {phase}"
            )

        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate
        )

        return optimizer

    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        phase,
        model_name,
        label_to_index,
        index_to_label,
        best_validation_loss,
        best_epoch,
        best_model_state,
        best_validation_accuracy=None,
        checkpoint_type="latest"
    ):
        """
        Saves a .pt checkpoint.

        latest:
            Saves the current model and optimizer so
            training can be resumed.

        best:
            Saves the best model for later testing
            and evaluation.
        """

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
                phase,

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
                    model.activation
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
            ] = self._copy_value(
                optimizer.state_dict()
            )

            file_name = (
                f"{model_name}_"
                f"{phase}_latest.pt"
            )

        else:

            file_name = (
                f"{model_name}_"
                f"{phase}_best.pt"
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

    def load_checkpoint(
        self,
        checkpoint_path,
        sample_batch,
        learning_rate
    ):
        """
        Restores a Transformer and optimizer from
        a previously saved checkpoint.

        Returns the model, optimizer, next epoch,
        and checkpoint information.
        """

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
            != configuration["patch_dim"]
        ):
            raise ValueError(
                "Dataset patch dimension does not "
                "match the saved checkpoint. "
                f"Dataset: "
                f"{sample_batch.shape[-1]}, "
                f"checkpoint: "
                f"{configuration['patch_dim']}."
            )

        model = Transformer(
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
            )
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            model.W_embedding.device != self.device
        ):
            raise RuntimeError(
                "Loaded model parameters are not "
                "on the expected device."
            )

        phase = checkpoint[
            "phase"
        ]

        optimizer = self._create_optimizer(
            model=model,
            phase=phase,
            learning_rate=learning_rate
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        next_epoch = (
            checkpoint["epoch"] + 1
        )

        return {
            "model":
                model,

            "optimizer":
                optimizer,

            "next_epoch":
                next_epoch,

            "phase":
                phase,

            "best_validation_loss":
                checkpoint[
                    "best_validation_loss"
                ],

            "best_validation_accuracy":
                checkpoint.get(
                    "best_validation_accuracy"
                ),

            "best_epoch":
                checkpoint[
                    "best_epoch"
                ],

            "best_model_state":
                checkpoint[
                    "best_model_state_dict"
                ],

            "label_to_index":
                checkpoint[
                    "label_to_index"
                ],

            "index_to_label":
                checkpoint[
                    "index_to_label"
                ],

            "checkpoint":
                checkpoint
        }

    def train_unsupervised_epoch(
        self,
        model,
        data_loader,
        optimizer
    ):
        """
        Runs one epoch of masked spectrogram
        reconstruction training.
        """

        total_loss = 0.0
        number_of_batches = 0

        for patches, _ in data_loader:

            patches = patches.to(
                device=self.device,
                dtype=torch.float32
            )

            optimizer.zero_grad()

            (
                loss,
                reconstructed_patches,
                mask,
                attention_weights
            ) = (
                model
                .unsupervised_reconstruction(
                    patches
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise ValueError(
                    "Unsupervised training loss "
                    "became NaN or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            number_of_batches += 1

        if number_of_batches == 0:
            raise ValueError(
                "Unsupervised training "
                "DataLoader contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        return average_loss

    def validate_unsupervised_epoch(
        self,
        model,
        data_loader
    ):
        """
        Runs one epoch of masked spectrogram
        reconstruction validation.
        """

        total_loss = 0.0
        number_of_batches = 0

        with torch.no_grad():

            for patches, _ in data_loader:

                patches = patches.to(
                    device=self.device,
                    dtype=torch.float32
                )

                (
                    loss,
                    reconstructed_patches,
                    mask,
                    attention_weights
                ) = (
                    model
                    .unsupervised_reconstruction(
                        patches
                    )
                )

                if not torch.isfinite(
                    loss
                ):
                    raise ValueError(
                        "Unsupervised validation "
                        "loss became NaN or infinite."
                    )

                total_loss += (
                    loss.item()
                )

                number_of_batches += 1

        if number_of_batches == 0:
            raise ValueError(
                "Unsupervised validation "
                "DataLoader contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        return average_loss

    def pretrain_unsupervised(
        self,
        model,
        training_loader,
        validation_loader,
        epochs,
        learning_rate,
        model_name,
        label_to_index,
        index_to_label,
        optimizer=None,
        start_epoch=1,
        best_validation_loss=float("inf"),
        best_epoch=None,
        best_model_state=None,
        patience=PATIENCE
    ):
        """
        Performs masked spectrogram reconstruction.

        The latest model is checkpointed every
        checkpoint_interval epochs.

        At the end of training, the model with the
        lowest validation loss is restored and
        saved separately.
        """

        if optimizer is None:

            optimizer = self._create_optimizer(
                model=model,
                phase="unsupervised",
                learning_rate=learning_rate
            )

        history = []

        latest_checkpoint_paths = []

        if best_model_state is None:

            best_model_state = (
                self.copy_model_state(
                    model
                )
            )

        patience_count = 0

        for epoch in tqdm(range(
            start_epoch,
            epochs + 1
        )):

            training_loss = (
                self.train_unsupervised_epoch(
                    model=model,
                    data_loader=training_loader,
                    optimizer=optimizer
                )
            )

            validation_loss = (
                self.validate_unsupervised_epoch(
                    model=model,
                    data_loader=validation_loader
                )
            )

            if (
                validation_loss < best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
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

                "validation_loss":
                    validation_loss,

                "best_validation_loss":
                    best_validation_loss,

                "best_epoch":
                    best_epoch
            })

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
                        phase="unsupervised",
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
                        best_epoch=(
                            best_epoch
                        ),
                        best_model_state=(
                            best_model_state
                        ),
                        checkpoint_type=(
                            "latest"
                        )
                    )
                )

                latest_checkpoint_paths.append(
                    checkpoint_path
                )

            if (
                patience_count >= patience
            ):
                    
                print(
                        "Validation loss early stop activated"
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
                phase="unsupervised",
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
                best_epoch=best_epoch,
                best_model_state=(
                    best_model_state
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

            "best_epoch":
                best_epoch,

            "latest_checkpoint_paths":
                latest_checkpoint_paths,

            "best_checkpoint_path":
                best_checkpoint_path
        }

    def train_supervised_epoch(
        self,
        model,
        data_loader,
        optimizer
    ):
        """
        Runs one epoch of supervised instrument
        classification training.
        """

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        number_of_batches = 0

        for patches, labels in data_loader:

            patches = patches.to(
                device=self.device,
                dtype=torch.float32
            )

            labels = labels.to(
                device=self.device,
                dtype=torch.long
            )

            optimizer.zero_grad()

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
                    "Supervised training loss "
                    "became NaN or infinite."
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            total_correct += (
                predicted_classes
                == labels
            ).sum().item()

            total_samples += (
                labels.shape[0]
            )

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
            / total_samples
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
        """
        Runs one epoch of supervised instrument
        classification validation.
        """

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        number_of_batches = 0

        with torch.no_grad():

            for patches, labels in data_loader:

                patches = patches.to(
                    device=self.device,
                    dtype=torch.float32
                )

                labels = labels.to(
                    device=self.device,
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
                        "Supervised validation "
                        "loss became NaN or infinite."
                    )

                total_loss += (
                    loss.item()
                )

                total_correct += (
                    predicted_classes
                    == labels
                ).sum().item()

                total_samples += (
                    labels.shape[0]
                )

                number_of_batches += 1

        if number_of_batches == 0:
            raise ValueError(
                "Supervised validation "
                "DataLoader contains no batches."
            )

        average_loss = (
            total_loss
            / number_of_batches
        )

        accuracy = (
            total_correct
            / total_samples
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
        optimizer=None,
        start_epoch=1,
        best_validation_loss=float("inf"),
        best_validation_accuracy=0.0,
        best_epoch=None,
        best_model_state=None,
        patience=PATIENCE
    ):
        """
        Performs supervised instrument
        classification training.

        The latest model is checkpointed every
        checkpoint_interval epochs.

        At the end, the model with the lowest
        validation loss is restored and saved.
        """

        if optimizer is None:

            optimizer = self._create_optimizer(
                model=model,
                phase="supervised",
                learning_rate=learning_rate
            )

        history = []

        latest_checkpoint_paths = []

        if best_model_state is None:

            best_model_state = (
                self.copy_model_state(
                    model
                )
            )

        patience_count = 0

        for epoch in tqdm(range(
            start_epoch,
            epochs + 1
        )):

            (
                training_loss,
                training_accuracy
            ) = (
                self.train_supervised_epoch(
                    model=model,
                    data_loader=training_loader,
                    optimizer=optimizer
                )
            )

            (
                validation_loss,
                validation_accuracy
            ) = (
                self.validate_supervised_epoch(
                    model=model,
                    data_loader=validation_loader
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
                        phase="supervised",
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
                        checkpoint_type=(
                            "latest"
                        )
                    )
                )

                latest_checkpoint_paths.append(
                    checkpoint_path
                )
                            
            if (
                patience_count >= patience
            ):
                                
                print(
                    "Validation loss early stop activated"
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
                phase="supervised",
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
                best_epoch=best_epoch,
                best_model_state=(
                    best_model_state
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

            "latest_checkpoint_paths":
                latest_checkpoint_paths,

            "best_checkpoint_path":
                best_checkpoint_path
        }

    def train_pretrained_and_fine_tuned_model(
        self,
        sample_batch,
        unsupervised_training_loader,
        unsupervised_validation_loader,
        supervised_training_loader,
        supervised_validation_loader,
        unsupervised_epochs,
        supervised_epochs,
        unsupervised_learning_rate,
        supervised_learning_rate,
        model_name,
        label_to_index,
        index_to_label
    ):
        """
        Creates a model, performs unsupervised
        pretraining, then supervised fine-tuning.
        """

        model = self.create_model(
            sample_batch
        )

        unsupervised_results = (
            self.pretrain_unsupervised(
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
                ),
                model_name=model_name,
                label_to_index=(
                    label_to_index
                ),
                index_to_label=(
                    index_to_label
                )
            )
        )

        model = (
            unsupervised_results[
                "model"
            ]
        )

        supervised_results = (
            self.fine_tune_supervised(
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
                model_name=model_name,
                label_to_index=(
                    label_to_index
                ),
                index_to_label=(
                    index_to_label
                )
            )
        )

        return {
            "model":
                supervised_results[
                    "model"
                ],

            "unsupervised_results":
                unsupervised_results,

            "supervised_results":
                supervised_results
        }

    def train_supervised_only_model(
        self,
        sample_batch,
        supervised_training_loader,
        supervised_validation_loader,
        epochs,
        learning_rate,
        model_name,
        label_to_index,
        index_to_label
    ):
        """
        Creates a new Transformer and trains it
        using only supervised classification.
        """

        model = self.create_model(
            sample_batch
        )

        results = (
            self.fine_tune_supervised(
                model=model,
                training_loader=(
                    supervised_training_loader
                ),
                validation_loader=(
                    supervised_validation_loader
                ),
                epochs=epochs,
                learning_rate=learning_rate,
                model_name=model_name,
                label_to_index=(
                    label_to_index
                ),
                index_to_label=(
                    index_to_label
                )
            )
        )

        return results

if __name__ == "__main__":

    # Create Datasets
    print("Creating datasets...")
    (
        supervised_training_dataset,
        supervised_validation_dataset,
        unsupervised_training_dataset,
        unsupervised_validation_dataset,
        _,
        label_to_index_val,
        label_to_index_test
    ) = create_datasets(
        training_path=(
            SLAKH2100_REDUX_16K_TRAIN
        ),
        validation_path=(
            SLAKH2100_REDUX_16K_VALIDATION
        ),
        test_path=(
            SLAKH2100_REDUX_16K_TEST
        ),
        data_representation=(
            DATA_REPRESENTATION
        ),
        set_limit=SET_LIMIT,
        maximum_tracks=MAXIMUM_TRACKS
    )

    # Retrieve model label mappings
    (
        label_to_index,
        index_to_label
    ) = (
        supervised_training_dataset
        .return_label_mappings()
    )

    num_classes = len(
        label_to_index
    )


    # Check datasets
    print(
        "\nDataset Information"
    )

    print("=" * 70)

    print(
        "Unsupervised training dataset size:",
        len(
            unsupervised_training_dataset
        )
    )

    print(
        "Unsupervised validation dataset size:",
        len(
            unsupervised_validation_dataset
        )
    )

    print(
        "Supervised training dataset size:",
        len(
            supervised_training_dataset
        )
    )

    print(
        "Supervised validation dataset size:",
        len(
            supervised_validation_dataset
        )
    )

    print(
        "Label to index mapping:",
        label_to_index
    )

    print(
        "Index to label mapping:",
        index_to_label
    )

    print(
        "Number of classes:",
        num_classes
    )

    print(
        "Training mapping matches "
        "validation mapping:",
        label_to_index
        == label_to_index_val
    )

    print(
        "Training mapping matches "
        "test mapping:",
        label_to_index
        == label_to_index_test
    )

    # DataLoaders
    unsupervised_training_loader = DataLoader(
        unsupervised_training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    unsupervised_validation_loader = DataLoader(
        unsupervised_validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    supervised_training_loader = DataLoader(
        supervised_training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    supervised_validation_loader = DataLoader(
        supervised_validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Get sample batch for Transformer initialization
    (
        sample_batch,
        _
    ) = next(
        iter(
            supervised_training_loader
        )
    )

    # Train and evaluate the model
    training_pipeline = TrainModelPipeline(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dims=HIDDEN_DIMS,
        num_encoder_layers=(
            NUM_ENCODER_LAYERS
        ),
        num_classes=num_classes,
        mask_ratio=MASK_RATIO,
        activation=ACTIVATION,
        checkpoint_interval=(
            CHECKPOINT_INTERVAL
        ),
        checkpoint_directory=(
            CHECKPOINT_DIRECTORY
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
            torch.cuda.get_device_name(0)
        )

    # Unsupervised pretraining + supervised fine-tuning
    # print("\n")
    # print("#" * 70)

    # print(
    #     "MODEL 1: UNSUPERVISED PRETRAINING "
    #     "FOLLOWED BY SUPERVISED FINE-TUNING"
    # )

    # print("#" * 70)


    # pretrained_results = (
    #     training_pipeline
    #     .train_pretrained_and_fine_tuned_model(
    #         sample_batch=sample_batch,
    #         unsupervised_training_loader=(
    #             unsupervised_training_loader
    #         ),
    #         unsupervised_validation_loader=(
    #             unsupervised_validation_loader
    #         ),
    #         supervised_training_loader=(
    #             supervised_training_loader
    #         ),
    #         supervised_validation_loader=(
    #             supervised_validation_loader
    #         ),
    #         unsupervised_epochs=(
    #             UNSUPERVISED_EPOCHS
    #         ),
    #         supervised_epochs=(
    #             SUPERVISED_EPOCHS
    #         ),
    #         unsupervised_learning_rate=(
    #             UNSUPERVISED_LEARNING_RATE
    #         ),
    #         supervised_learning_rate=(
    #             SUPERVISED_LEARNING_RATE
    #         ),
    #         model_name=(
    #             PRETRAINED_MODEL_NAME
    #         ),
    #         label_to_index=(
    #             label_to_index
    #         ),
    #         index_to_label=(
    #             index_to_label
    #         )
    #     )
    # )

    # # Retrieve trained model
    # pretrained_fine_tuned_model = (
    #     pretrained_results[
    #         "model"
    #     ]
    # )

    # # Print unsupervised training history
    # print(
    #     "\nUnsupervised Pretraining"
    # )

    # print("=" * 70)

    # for epoch_results in (
    #     pretrained_results[
    #         "unsupervised_results"
    #     ]["history"]
    # ):

    #     print(
    #         f"Epoch "
    #         f"{epoch_results['epoch']} | "
    #         f"Training loss: "
    #         f"{epoch_results['training_loss']:.6f} | "
    #         f"Validation loss: "
    #         f"{epoch_results['validation_loss']:.6f}"
    #     )


    # print(
    #     "\nBest unsupervised validation loss:",
    #     pretrained_results[
    #         "unsupervised_results"
    #     ][
    #         "best_validation_loss"
    #     ]
    # )

    # print(
    #     "Best unsupervised epoch:",
    #     pretrained_results[
    #         "unsupervised_results"
    #     ][
    #         "best_epoch"
    #     ]
    # )

    # # Print supervised fine-tuning history
    # print(
    #     "\nSupervised Fine-Tuning"
    # )

    # print("=" * 70)

    # for epoch_results in (
    #     pretrained_results[
    #         "supervised_results"
    #     ]["history"]
    # ):

    #     print(
    #         f"Epoch "
    #         f"{epoch_results['epoch']} | "
    #         f"Training loss: "
    #         f"{epoch_results['training_loss']:.6f} | "
    #         f"Training accuracy: "
    #         f"{epoch_results['training_accuracy'] * 100:.2f}% | "
    #         f"Validation loss: "
    #         f"{epoch_results['validation_loss']:.6f} | "
    #         f"Validation accuracy: "
    #         f"{epoch_results['validation_accuracy'] * 100:.2f}%"
    #     )


    # print(
    #     "\nBest supervised validation loss:",
    #     pretrained_results[
    #         "supervised_results"
    #     ][
    #         "best_validation_loss"
    #     ]
    # )

    # print(
    #     "Best supervised validation accuracy:",
    #     (
    #         pretrained_results[
    #             "supervised_results"
    #         ][
    #             "best_validation_accuracy"
    #         ]
    #         * 100
    #     )
    # )

    # print(
    #     "Best supervised epoch:",
    #     pretrained_results[
    #         "supervised_results"
    #     ][
    #         "best_epoch"
    #     ]
    # )

    # Supervised-only training
    print("\n")
    print("#" * 70)

    print(
        "MODEL 2: SUPERVISED-ONLY TRAINING"
    )

    print("#" * 70)


    supervised_only_results = (
        training_pipeline
        .train_supervised_only_model(
            sample_batch=sample_batch,
            supervised_training_loader=(
                supervised_training_loader
            ),
            supervised_validation_loader=(
                supervised_validation_loader
            ),
            epochs=SUPERVISED_EPOCHS,
            learning_rate=(
                SUPERVISED_LEARNING_RATE
            ),
            model_name=(
                SUPERVISED_ONLY_MODEL_NAME
            ),
            label_to_index=(
                label_to_index
            ),
            index_to_label=(
                index_to_label
            )
        )
    )

    # Retrieve supervised-only model
    supervised_only_model = (
        supervised_only_results[
            "model"
        ]
    )

    # Print supervised-only training history
    print(
        "\nSupervised-Only Training"
    )

    print("=" * 70)

    for epoch_results in (
        supervised_only_results[
            "history"
        ]
    ):

        print(
            f"Epoch "
            f"{epoch_results['epoch']} | "
            f"Training loss: "
            f"{epoch_results['training_loss']:.6f} | "
            f"Training accuracy: "
            f"{epoch_results['training_accuracy'] * 100:.2f}% | "
            f"Validation loss: "
            f"{epoch_results['validation_loss']:.6f} | "
            f"Validation accuracy: "
            f"{epoch_results['validation_accuracy'] * 100:.2f}%"
        )


    print(
        "\nBest supervised-only "
        "validation loss:",
        supervised_only_results[
            "best_validation_loss"
        ]
    )

    print(
        "Best supervised-only "
        "validation accuracy:",
        (
            supervised_only_results[
                "best_validation_accuracy"
            ]
            * 100
        )
    )

    print(
        "Best supervised-only epoch:",
        supervised_only_results[
            "best_epoch"
        ]
    )