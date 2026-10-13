import warnings
from pathlib import Path

import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)

from data_processing_pipeline import (
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram
)

from multi_label_data_processing import (
    MULTI_LABEL_MAPPING,
    INDEX_TO_MULTI_LABEL,
    NUM_CLASSES,
    STEM_ACTIVE_LOUDNESS_THRESHOLD_DB,
    MIX_LOUDNESS_THRESHOLD_DB,
    build_multilabel_sample_index,
    load_multilabel_mix_sample,
    multi_hot_to_labels
)


# ---------------------------------------------------------------------
# Ignore selected Librosa warnings
# ---------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed.*",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load.*",
    category=FutureWarning
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BATCH_SIZE = 16
SHUFFLE_TRAINING_DATA = True

SET_TRACK_LIMIT = False
MAXIMUM_TRACKS = 20

NUM_WORKERS = 0
PIN_MEMORY = False


# ---------------------------------------------------------------------
# Slakh2100 Redux 16k paths
# ---------------------------------------------------------------------

SLAKH2100_REDUX_16K_TEST = Path(
    "C:/Uni/YearProject/datasets/"
    "slakh2100_redux_16k/test"
)

SLAKH2100_REDUX_16K_TRAIN = Path(
    "C:/Uni/YearProject/datasets/"
    "slakh2100_redux_16k/train"
)

SLAKH2100_REDUX_16K_VALIDATION = Path(
    "C:/Uni/YearProject/datasets/"
    "slakh2100_redux_16k/validation"
)


# ---------------------------------------------------------------------
# Multi-label dataset class
# ---------------------------------------------------------------------

class MultiLabelAudioDataset(
    Dataset
):
    """
    Stores lightweight metadata for valid multi-label
    mix.flac 5-second samples.

    The sample index is created before training by
    checking which stems are active in each aligned
    5-second interval.

    Spectrograms and patch tensors are generated
    lazily when __getitem__ is called.

    Each returned label is a float32 multi-hot tensor
    with shape:

        [number_of_classes]
    """

    def __init__(
        self,
        sample_index,
        processing_stats,
        data_representation=(
            convert_to_mel_spectrogram
        ),
        label_to_index=(
            MULTI_LABEL_MAPPING
        )
    ):
        self.samples = list(
            sample_index
        )

        self.processing_stats = (
            processing_stats
        )

        self.data_representation = (
            data_representation
        )

        self.label_to_index = dict(
            label_to_index
        )

        self.index_to_label = {
            index: label
            for label, index
            in self.label_to_index.items()
        }

        self.num_classes = len(
            self.label_to_index
        )

        self._validate_label_mapping()

        self._validate_sample_index()


    def _validate_label_mapping(
        self
    ):
        """
        Verifies that the multi-label mapping uses
        consecutive indices beginning at zero.
        """

        expected_indices = list(
            range(
                len(
                    self.label_to_index
                )
            )
        )

        actual_indices = sorted(
            self.label_to_index.values()
        )

        if (
            actual_indices
            != expected_indices
        ):
            raise ValueError(
                "Label mapping indices must be "
                "consecutive and begin at zero."
            )

        if (
            self.num_classes
            != NUM_CLASSES
        ):
            raise ValueError(
                "Dataset class count does not match "
                "the configured multi-label class count. "
                f"Dataset: {self.num_classes}, "
                f"configured: {NUM_CLASSES}."
            )


    def _validate_sample_index(
        self
    ):
        """
        Verifies that each indexed sample contains
        the metadata required for lazy loading.
        """

        required_keys = {
            "track",
            "mix_path",
            "start_sample",
            "end_sample",
            "sample_rate",
            "labels",
            "label_indices",
            "label_vector"
        }

        for sample_index, sample in enumerate(
            self.samples
        ):

            missing_keys = (
                required_keys
                - set(
                    sample.keys()
                )
            )

            if missing_keys:
                raise KeyError(
                    f"Sample {sample_index} is missing "
                    f"required metadata keys: "
                    f"{sorted(missing_keys)}"
                )

            label_vector = (
                sample[
                    "label_vector"
                ]
            )

            if not isinstance(
                label_vector,
                torch.Tensor
            ):
                raise TypeError(
                    f"Sample {sample_index} label_vector "
                    "must be a PyTorch tensor."
                )

            if label_vector.ndim != 1:
                raise ValueError(
                    f"Sample {sample_index} label_vector "
                    "must be one-dimensional."
                )

            if (
                label_vector.shape[0]
                != self.num_classes
            ):
                raise ValueError(
                    f"Sample {sample_index} label_vector "
                    "has an incorrect class dimension. "
                    f"Expected {self.num_classes}, "
                    f"received "
                    f"{label_vector.shape[0]}."
                )


    def return_label_mappings(
        self
    ):
        """
        Returns label-to-index and index-to-label
        mappings.
        """

        return (
            dict(
                self.label_to_index
            ),
            dict(
                self.index_to_label
            )
        )


    def return_processing_stats(
        self
    ):
        """
        Returns the statistics created while building
        the multi-label sample index.
        """

        return (
            self.processing_stats
        )


    def return_sample_information(
        self,
        index
    ):
        """
        Returns the lightweight metadata for one
        indexed sample without generating patches.
        """

        return (
            self.samples[
                index
            ]
        )


    def __len__(
        self
    ):
        """
        Returns the number of valid indexed
        multi-label samples.
        """

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):
        """
        Lazily loads one 5-second mix.flac interval,
        converts it into Transformer patches and
        returns its multi-hot target vector.
        """

        sample_information = (
            self.samples[
                index
            ]
        )

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

        if not isinstance(
            patches,
            torch.Tensor
        ):
            raise TypeError(
                "Processed sample must be "
                "a PyTorch tensor."
            )

        if patches.ndim != 2:
            raise ValueError(
                "Expected patch tensor shape "
                "[number_of_patches, patch_features], "
                f"but received {patches.shape}."
            )

        if not isinstance(
            label_vector,
            torch.Tensor
        ):
            raise TypeError(
                "Multi-label target must be "
                "a PyTorch tensor."
            )

        if label_vector.ndim != 1:
            raise ValueError(
                "Expected target shape "
                "[number_of_classes], "
                f"but received "
                f"{label_vector.shape}."
            )

        if (
            label_vector.shape[0]
            != self.num_classes
        ):
            raise ValueError(
                "Multi-label target class dimension "
                "does not match the dataset mapping."
            )

        sample = (
            patches.to(
                dtype=torch.float32
            )
        )

        label = (
            label_vector.to(
                dtype=torch.float32
            )
        )

        return (
            sample,
            label
        )


# ---------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------

def create_multi_label_dataset(
    dataset_path,
    data_representation=(
        convert_to_mel_spectrogram
    ),
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    show_progress=True
):
    """
    Creates one MultiLabelAudioDataset from a Slakh
    train, validation or test split.

    The function first scans each track and creates
    aligned 5-second labels from its stems. The
    spectrogram itself is still generated lazily by
    the Dataset when a sample is requested.
    """

    dataset_path = Path(
        dataset_path
    )

    (
        sample_index,
        processing_stats
    ) = (
        build_multilabel_sample_index(
            dataset_path=(
                dataset_path
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            label_to_index=(
                MULTI_LABEL_MAPPING
            ),
            show_progress=(
                show_progress
            )
        )
    )

    dataset = (
        MultiLabelAudioDataset(
            sample_index=(
                sample_index
            ),
            processing_stats=(
                processing_stats
            ),
            data_representation=(
                data_representation
            ),
            label_to_index=(
                MULTI_LABEL_MAPPING
            )
        )
    )

    return dataset


def create_train_val_datasets(
    training_path,
    validation_path,
    data_representation=(
        convert_to_mel_spectrogram
    ),
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    show_progress=True
):
    """
    Creates multi-label training and validation
    datasets using the official Slakh splits.
    """

    training_dataset = (
        create_multi_label_dataset(
            dataset_path=(
                training_path
            ),
            data_representation=(
                data_representation
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            show_progress=(
                show_progress
            )
        )
    )

    validation_dataset = (
        create_multi_label_dataset(
            dataset_path=(
                validation_path
            ),
            data_representation=(
                data_representation
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            show_progress=(
                show_progress
            )
        )
    )

    _validate_dataset_mappings(
        training_dataset,
        validation_dataset
    )

    return (
        training_dataset,
        validation_dataset
    )


def create_test_dataset(
    test_path,
    data_representation=(
        convert_to_mel_spectrogram
    ),
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    show_progress=True
):
    """
    Creates the held-out multi-label test dataset.
    """

    test_dataset = (
        create_multi_label_dataset(
            dataset_path=(
                test_path
            ),
            data_representation=(
                data_representation
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            show_progress=(
                show_progress
            )
        )
    )

    return test_dataset


def create_datasets(
    training_path,
    validation_path,
    test_path,
    data_representation=(
        convert_to_mel_spectrogram
    ),
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    show_progress=True
):
    """
    Creates the complete multi-label training,
    validation and held-out test datasets.
    """

    (
        training_dataset,
        validation_dataset
    ) = (
        create_train_val_datasets(
            training_path=(
                training_path
            ),
            validation_path=(
                validation_path
            ),
            data_representation=(
                data_representation
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            show_progress=(
                show_progress
            )
        )
    )

    test_dataset = (
        create_test_dataset(
            test_path=(
                test_path
            ),
            data_representation=(
                data_representation
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            ),
            mix_loudness_threshold=(
                mix_loudness_threshold
            ),
            stem_loudness_threshold=(
                stem_loudness_threshold
            ),
            show_progress=(
                show_progress
            )
        )
    )

    _validate_dataset_mappings(
        training_dataset,
        validation_dataset,
        test_dataset
    )

    return (
        training_dataset,
        validation_dataset,
        test_dataset
    )


# ---------------------------------------------------------------------
# Mapping validation
# ---------------------------------------------------------------------

def _validate_dataset_mappings(
    *datasets
):
    """
    Ensures that every dataset uses exactly the same
    multi-label class mapping.
    """

    if len(datasets) < 2:
        return

    reference_mapping = (
        datasets[0]
        .return_label_mappings()[0]
    )

    for dataset in datasets[1:]:

        current_mapping = (
            dataset
            .return_label_mappings()[0]
        )

        if (
            current_mapping
            != reference_mapping
        ):
            raise ValueError(
                "Multi-label mappings do not match "
                "between dataset splits."
            )


# ---------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------

def create_data_loader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
):
    """
    Creates one PyTorch DataLoader.
    """

    if len(dataset) == 0:
        raise ValueError(
            "Cannot create a DataLoader from "
            "an empty dataset."
        )

    data_loader = (
        DataLoader(
            dataset,
            batch_size=(
                batch_size
            ),
            shuffle=(
                shuffle
            ),
            num_workers=(
                num_workers
            ),
            pin_memory=(
                pin_memory
            )
        )
    )

    return data_loader


def create_train_val_data_loaders(
    training_dataset,
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle_training=(
        SHUFFLE_TRAINING_DATA
    ),
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
):
    """
    Creates training and validation DataLoaders.
    """

    training_loader = (
        create_data_loader(
            dataset=(
                training_dataset
            ),
            batch_size=(
                batch_size
            ),
            shuffle=(
                shuffle_training
            ),
            num_workers=(
                num_workers
            ),
            pin_memory=(
                pin_memory
            )
        )
    )

    validation_loader = (
        create_data_loader(
            dataset=(
                validation_dataset
            ),
            batch_size=(
                batch_size
            ),
            shuffle=False,
            num_workers=(
                num_workers
            ),
            pin_memory=(
                pin_memory
            )
        )
    )

    return (
        training_loader,
        validation_loader
    )


def create_all_data_loaders(
    training_dataset,
    validation_dataset,
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle_training=(
        SHUFFLE_TRAINING_DATA
    ),
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
):
    """
    Creates training, validation and test DataLoaders.
    """

    (
        training_loader,
        validation_loader
    ) = (
        create_train_val_data_loaders(
            training_dataset=(
                training_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            batch_size=(
                batch_size
            ),
            shuffle_training=(
                shuffle_training
            ),
            num_workers=(
                num_workers
            ),
            pin_memory=(
                pin_memory
            )
        )
    )

    test_loader = (
        create_data_loader(
            dataset=(
                test_dataset
            ),
            batch_size=(
                batch_size
            ),
            shuffle=False,
            num_workers=(
                num_workers
            ),
            pin_memory=(
                pin_memory
            )
        )
    )

    return (
        training_loader,
        validation_loader,
        test_loader
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def check_multi_label_dataset(
    dataset,
    dataset_name
):
    """
    Prints useful information about one multi-label
    dataset and checks the first sample.
    """

    print(
        f"\n{dataset_name}"
    )

    print(
        "=" * 70
    )

    stats = (
        dataset
        .return_processing_stats()
    )

    print(
        f"Number of samples: "
        f"{len(dataset)}"
    )

    print(
        f"Number of classes: "
        f"{dataset.num_classes}"
    )

    print(
        f"Tracks found: "
        f"{stats['tracks_found']}"
    )

    print(
        f"Valid tracks: "
        f"{stats['valid_tracks']}"
    )

    print(
        f"Invalid tracks: "
        f"{stats['invalid_tracks']}"
    )

    print(
        f"Total 5-second samples: "
        f"{stats['total_sub_samples']}"
    )

    print(
        f"Valid labelled samples: "
        f"{stats['valid_sub_samples']}"
    )

    print(
        f"Quiet mixture samples: "
        f"{stats['quiet_mix_sub_samples']}"
    )

    print(
        f"Samples with no active labels: "
        f"{stats['no_active_label_sub_samples']}"
    )

    print(
        "\nPer-label active sample counts"
    )

    print(
        "-" * 70
    )

    for label, count in (
        stats[
            "sample_label_balance"
        ].items()
    ):

        print(
            f"{label:<25}"
            f"{count:>10}"
        )

    if len(dataset) == 0:

        print(
            "\nThe dataset contains "
            "no valid samples."
        )

        return

    (
        first_sample,
        first_label
    ) = dataset[0]

    first_information = (
        dataset
        .return_sample_information(
            0
        )
    )

    print(
        "\nFirst sample"
    )

    print(
        "-" * 70
    )

    print(
        f"Track: "
        f"{first_information['track']}"
    )

    print(
        f"Time: "
        f"{first_information['start_time']:.2f} "
        f"- "
        f"{first_information['end_time']:.2f} "
        f"seconds"
    )

    print(
        f"First sample type: "
        f"{type(first_sample)}"
    )

    print(
        f"First sample shape: "
        f"{first_sample.shape}"
    )

    print(
        f"First sample dtype: "
        f"{first_sample.dtype}"
    )

    print(
        f"First label shape: "
        f"{first_label.shape}"
    )

    print(
        f"First label dtype: "
        f"{first_label.dtype}"
    )

    print(
        f"First label vector: "
        f"{first_label}"
    )

    print(
        f"Active labels: "
        f"{multi_hot_to_labels(first_label)}"
    )

    print(
        f"Number of active classes: "
        f"{int(first_label.sum().item())}"
    )

    print(
        f"Number of patches: "
        f"{first_sample.shape[0]}"
    )

    print(
        f"Values per patch: "
        f"{first_sample.shape[1]}"
    )


def check_data_loader(
    data_loader,
    loader_name
):
    """
    Prints the shape and dtype of one multi-label
    DataLoader batch.
    """

    if (
        len(
            data_loader.dataset
        )
        == 0
    ):

        print(
            f"\n{loader_name} "
            f"contains no samples."
        )

        return

    (
        sample_batch,
        label_batch
    ) = next(
        iter(
            data_loader
        )
    )

    print(
        f"\n{loader_name}"
    )

    print(
        "=" * 70
    )

    print(
        f"Sample batch shape: "
        f"{sample_batch.shape}"
    )

    print(
        f"Sample batch dtype: "
        f"{sample_batch.dtype}"
    )

    print(
        f"Label batch shape: "
        f"{label_batch.shape}"
    )

    print(
        f"Label batch dtype: "
        f"{label_batch.dtype}"
    )

    print(
        f"First multi-hot target: "
        f"{label_batch[0]}"
    )

    print(
        f"First target labels: "
        f"{multi_hot_to_labels(label_batch[0])}"
    )

    print(
        f"Active classes in first target: "
        f"{int(label_batch[0].sum().item())}"
    )


def compare_dataset_statistics(
    datasets
):
    """
    Prints a compact comparison of train,
    validation and test dataset statistics.

    datasets should be a dictionary such as:

        {
            "Train": training_dataset,
            "Validation": validation_dataset,
            "Test": test_dataset
        }
    """

    print(
        "\nDataset Statistics Comparison"
    )

    print(
        "=" * 90
    )

    header = (
        f"{'Statistic':<30}"
    )

    for dataset_name in datasets:

        header += (
            f"{dataset_name:>18}"
        )

    print(
        header
    )

    print(
        "-" * 90
    )

    statistic_keys = [
        "tracks_found",
        "valid_tracks",
        "invalid_tracks",
        "total_sub_samples",
        "valid_sub_samples",
        "quiet_mix_sub_samples",
        "no_active_label_sub_samples"
    ]

    for statistic in statistic_keys:

        row = (
            f"{statistic:<30}"
        )

        for dataset in (
            datasets.values()
        ):

            stats = (
                dataset
                .return_processing_stats()
            )

            row += (
                f"{stats[statistic]:>18}"
            )

        print(
            row
        )

    row = (
        f"{'dataset_samples':<30}"
    )

    for dataset in (
        datasets.values()
    ):

        row += (
            f"{len(dataset):>18}"
        )

    print(
        row
    )

    print(
        "\nActive Sample Count Per Label"
    )

    print(
        "=" * 90
    )

    header = (
        f"{'Label':<30}"
    )

    for dataset_name in datasets:

        header += (
            f"{dataset_name:>18}"
        )

    print(
        header
    )

    print(
        "-" * 90
    )

    for label in (
        MULTI_LABEL_MAPPING
    ):

        row = (
            f"{label:<30}"
        )

        for dataset in (
            datasets.values()
        ):

            stats = (
                dataset
                .return_processing_stats()
            )

            count = (
                stats[
                    "sample_label_balance"
                ].get(
                    label,
                    0
                )
            )

            row += (
                f"{count:>18}"
            )

        print(
            row
        )


# ---------------------------------------------------------------------
# Basic test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Keep this small while checking the multi-label
    # pipeline. Change to False for the full dataset.
    TEST_WITH_TRACK_LIMIT = False
    TEST_MAXIMUM_TRACKS = 2

    DATA_REPRESENTATION = (
        convert_to_mel_spectrogram
    )

    (
        training_dataset,
        validation_dataset,
        test_dataset
    ) = (
        create_datasets(
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
            set_limit=(
                TEST_WITH_TRACK_LIMIT
            ),
            maximum_tracks=(
                TEST_MAXIMUM_TRACKS
            ),
            show_progress=True
        )
    )

    datasets = {
        "Train":
            training_dataset,

        "Validation":
            validation_dataset,

        "Test":
            test_dataset
    }

    compare_dataset_statistics(
        datasets
    )

    check_multi_label_dataset(
        training_dataset,
        "Multi-Label Training Dataset"
    )

    check_multi_label_dataset(
        validation_dataset,
        "Multi-Label Validation Dataset"
    )

    check_multi_label_dataset(
        test_dataset,
        "Multi-Label Test Dataset"
    )

    (
        training_loader,
        validation_loader,
        test_loader
    ) = (
        create_all_data_loaders(
            training_dataset=(
                training_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            test_dataset=(
                test_dataset
            ),
            batch_size=(
                BATCH_SIZE
            )
        )
    )

    check_data_loader(
        training_loader,
        "Multi-Label Training Batch"
    )

    check_data_loader(
        validation_loader,
        "Multi-Label Validation Batch"
    )

    check_data_loader(
        test_loader,
        "Multi-Label Test Batch"
    )