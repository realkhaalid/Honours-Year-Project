import warnings
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import (
    StratifiedGroupKFold
)

from data_processing_pipeline import (
    find_audio_files,
    load_and_validate_audio,
    cut_audio_to_consistent_length,
    find_valid_subsamples,
    load_audio_subsample,
    process_audio_subsample,
    convert_to_mel_spectrogram,
    convert_to_stft_spectrogram,
    convert_to_cqt_spectrogram
)


# Ignore selected Librosa warnings
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


# Config
BATCH_SIZE = 16
SHUFFLE_TRAINING_DATA = True

SET_TRACK_LIMIT = False
MAXIMUM_TRACKS = 20


# Slakh2100 Redux 16k
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


# Supervised dataset class
class SupervisedAudioDataset(
    Dataset
):
    """
    Stores metadata for valid labelled audio
    sub-samples.

    Spectrograms and patch tensors are generated
    lazily when __getitem__ is called.
    """

    def __init__(
        self,
        labelled_audio_files,
        label_to_index,
        data_representation=(
            convert_to_mel_spectrogram
        )
    ):
        self.samples = []
        self.labels = []

        self.label_to_index = dict(
            label_to_index
        )

        self.index_to_label = {
            index: label
            for label, index
            in self.label_to_index.items()
        }

        self.data_representation = (
            data_representation
        )

        self.processing_stats = (
            self._create_processing_stats(
                labelled_audio_files
            )
        )

        self._process_audio_files(
            labelled_audio_files
        )


    def _create_processing_stats(
        self,
        labelled_audio_files
    ):
        """
        Creates the initial dataset statistics.
        """

        label_source_counts = {
            label: 0
            for label
            in self.label_to_index
        }

        label_valid_sample_counts = {
            label: 0
            for label
            in self.label_to_index
        }

        label_quiet_sample_counts = {
            label: 0
            for label
            in self.label_to_index
        }

        return {
            "source_files":
                len(
                    labelled_audio_files
                ),

            "valid_source_files":
                0,

            "invalid_source_files":
                0,

            "total_sub_samples":
                0,

            "valid_sub_samples":
                0,

            "quiet_sub_samples":
                0,

            "source_label_balance":
                label_source_counts,

            "sample_label_balance":
                label_valid_sample_counts,

            "quiet_sample_balance":
                label_quiet_sample_counts,

            "invalid_files":
                []
        }


    def _update_source_label_count(
        self,
        instrument_label
    ):
        """
        Updates the number of source stems
        for a label.
        """

        if (
            instrument_label
            not in self.label_to_index
        ):

            raise KeyError(
                f"Unknown instrument label: "
                f"{instrument_label}"
            )

        self.processing_stats[
            "source_label_balance"
        ][instrument_label] += 1


    def _process_audio_files(
        self,
        labelled_audio_files
    ):
        """
        Scans every labelled source stem and
        stores metadata for valid sub-samples.

        No spectrogram or patch tensors are
        permanently stored here.
        """

        for (
            audio_file,
            instrument_label
        ) in tqdm(
            labelled_audio_files
        ):

            self._update_source_label_count(
                instrument_label
            )

            try:

                audio, sample_rate = (
                    load_and_validate_audio(
                        audio_file
                    )
                )

                audio_cut = (
                    cut_audio_to_consistent_length(
                        audio,
                        sample_rate
                    )
                )

                (
                    valid_start_samples,
                    statistics
                ) = (
                    find_valid_subsamples(
                        audio_cut,
                        sample_rate
                    )
                )

                self._store_valid_samples(
                    audio_file=audio_file,
                    sample_rate=sample_rate,
                    valid_start_samples=(
                        valid_start_samples
                    ),
                    instrument_label=(
                        instrument_label
                    )
                )

                self._update_processing_stats(
                    statistics,
                    instrument_label
                )

                self.processing_stats[
                    "valid_source_files"
                ] += 1

            except (
                ValueError,
                TypeError,
                KeyError,
                OSError
            ) as error:

                self.processing_stats[
                    "invalid_source_files"
                ] += 1

                self.processing_stats[
                    "invalid_files"
                ].append(
                    {
                        "file":
                            str(
                                audio_file
                            ),

                        "label":
                            instrument_label,

                        "reason":
                            str(
                                error
                            )
                    }
                )


    def _store_valid_samples(
        self,
        audio_file,
        sample_rate,
        valid_start_samples,
        instrument_label
    ):
        """
        Stores lightweight metadata for each
        valid audio sub-sample.
        """

        if (
            instrument_label
            not in self.label_to_index
        ):

            raise KeyError(
                f"Unknown instrument label: "
                f"{instrument_label}"
            )

        label_index = (
            self.label_to_index[
                instrument_label
            ]
        )

        for start_sample in (
            valid_start_samples
        ):

            self.samples.append(
                {
                    "audio_file":
                        Path(
                            audio_file
                        ),

                    "start_sample":
                        int(
                            start_sample
                        ),

                    "sample_rate":
                        int(
                            sample_rate
                        )
                }
            )

            self.labels.append(
                label_index
            )


    def _update_processing_stats(
        self,
        statistics,
        instrument_label
    ):
        """
        Updates overall and label-specific
        statistics.
        """

        total_samples = (
            statistics[
                "total_sub_samples"
            ]
        )

        valid_samples = (
            statistics[
                "valid_sub_samples"
            ]
        )

        quiet_samples = (
            statistics[
                "quiet_sub_samples"
            ]
        )

        self.processing_stats[
            "total_sub_samples"
        ] += total_samples

        self.processing_stats[
            "valid_sub_samples"
        ] += valid_samples

        self.processing_stats[
            "quiet_sub_samples"
        ] += quiet_samples

        self.processing_stats[
            "sample_label_balance"
        ][instrument_label] += (
            valid_samples
        )

        self.processing_stats[
            "quiet_sample_balance"
        ][instrument_label] += (
            quiet_samples
        )


    def return_label_mappings(
        self
    ):
        """
        Returns the label-to-index and
        index-to-label mappings.
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
        Returns dataset processing statistics.
        """

        return (
            self.processing_stats
        )


    def __len__(
        self
    ):
        """
        Returns the number of valid indexed
        audio sub-samples.
        """

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):
        """
        Lazily loads and processes one valid
        audio sub-sample.
        """

        sample_information = (
            self.samples[
                index
            ]
        )

        (
            audio,
            sample_rate
        ) = (
            load_audio_subsample(
                file_path=(
                    sample_information[
                        "audio_file"
                    ]
                ),
                start_sample=(
                    sample_information[
                        "start_sample"
                    ]
                ),
                sample_rate=(
                    sample_information[
                        "sample_rate"
                    ]
                )
            )
        )

        patches = (
            process_audio_subsample(
                audio,
                sample_rate,
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

        if patches.ndim != 3:

            raise ValueError(
                "Expected patch tensor shape "
                "[1, number_of_patches, "
                "patch_features], but received "
                f"{patches.shape}."
            )

        if patches.shape[0] != 1:

            raise ValueError(
                "The first patch tensor "
                "dimension must be 1."
            )

        sample = (
            patches
            .squeeze(0)
            .to(
                dtype=torch.float32
            )
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return (
            sample,
            label
        )


# Collect labelled files
def collect_supervised_audio_files(
    dataset_path,
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS
):
    """
    Retrieves labelled stems.
    """

    labelled_audio_files = (
        find_audio_files(
            dataset_path=(
                dataset_path
            ),
            set_limit=set_limit,
            maximum_files=(
                maximum_tracks
            ),
            supervised=True
        )
    )

    return (
        labelled_audio_files
    )


# Create label mapping
def create_label_mapping(
    labelled_audio_files
):
    """
    Creates the shared instrument label mapping.
    """

    unique_labels = sorted({
        label
        for _, label
        in labelled_audio_files
    })

    if not unique_labels:

        raise ValueError(
            "No instrument labels "
            "were found."
        )

    label_to_index = {
        label: index
        for index, label
        in enumerate(
            unique_labels
        )
    }

    return (
        label_to_index
    )


# Split labelled files
def split_training_files(
    labelled_audio_files,
    random_seed=42
):
    """
    Splits labelled training files into
    unsupervised and supervised groups.
    """

    labels = [
        label
        for _, label
        in labelled_audio_files
    ]

    track_groups = [
        audio_file.parent.parent.name
        for audio_file, _
        in labelled_audio_files
    ]

    splitter = (
        StratifiedGroupKFold(
            n_splits=2,
            shuffle=True,
            random_state=(
                random_seed
            )
        )
    )

    (
        unsupervised_indices,
        supervised_indices
    ) = next(
        splitter.split(
            labelled_audio_files,
            labels,
            groups=(
                track_groups
            )
        )
    )

    unsupervised_files = [
        labelled_audio_files[
            index
        ]
        for index
        in unsupervised_indices
    ]

    supervised_files = [
        labelled_audio_files[
            index
        ]
        for index
        in supervised_indices
    ]

    return (
        unsupervised_files,
        supervised_files
    )


# Create datasets
def create_datasets(
    training_path,
    validation_path,
    test_path,
    data_representation=(
        convert_to_mel_spectrogram
    ),
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS
):
    """
    Creates datasets using the official
    Slakh train, validation, and test
    partitions.
    """

    training_files = (
        collect_supervised_audio_files(
            training_path,
            set_limit=set_limit,
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    (
        unsupervised_training_files,
        supervised_training_files
    ) = split_training_files(
        training_files
    )

    validation_files = (
        collect_supervised_audio_files(
            validation_path,
            set_limit=set_limit,
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    (
        unsupervised_validation_files,
        supervised_validation_files
    ) = split_training_files(
        validation_files
    )

    test_files = (
        collect_supervised_audio_files(
            test_path,
            set_limit=set_limit,
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    label_to_index = (
        create_label_mapping(
            training_files
        )
    )

    label_to_index_val = (
        create_label_mapping(
            validation_files
        )
    )

    label_to_index_test = (
        create_label_mapping(
            test_files
        )
    )

    supervised_training_dataset = (
        SupervisedAudioDataset(
            supervised_training_files,
            label_to_index,
            data_representation=(
                data_representation
            )
        )
    )

    supervised_validation_dataset = (
        SupervisedAudioDataset(
            supervised_validation_files,
            label_to_index,
            data_representation=(
                data_representation
            )
        )
    )

    unsupervised_training_dataset = (
        SupervisedAudioDataset(
            unsupervised_training_files,
            label_to_index,
            data_representation=(
                data_representation
            )
        )
    )

    unsupervised_validation_dataset = (
        SupervisedAudioDataset(
            unsupervised_validation_files,
            label_to_index,
            data_representation=(
                data_representation
            )
        )
    )

    test_dataset = (
        SupervisedAudioDataset(
            test_files,
            label_to_index,
            data_representation=(
                data_representation
            )
        )
    )

    return (
        supervised_training_dataset,
        supervised_validation_dataset,
        unsupervised_training_dataset,
        unsupervised_validation_dataset,
        test_dataset,
        label_to_index_val,
        label_to_index_test
    )


# Check dataset information
def check_supervised_dataset(
    dataset,
    dataset_name
):
    """
    Prints information about a dataset.
    """

    print(
        f"\n{dataset_name}"
    )

    print("=" * 60)

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
        f"{len(dataset.label_to_index)}"
    )

    print(
        f"Source files: "
        f"{stats['source_files']}"
    )

    print(
        f"Valid source files: "
        f"{stats['valid_source_files']}"
    )

    print(
        f"Invalid source files: "
        f"{stats['invalid_source_files']}"
    )

    print(
        f"Total sub-samples: "
        f"{stats['total_sub_samples']}"
    )

    print(
        f"Valid sub-samples: "
        f"{stats['valid_sub_samples']}"
    )

    print(
        f"Quiet sub-samples: "
        f"{stats['quiet_sub_samples']}"
    )

    print(
        "\nSource-file label balance"
    )

    print("-" * 60)

    for label, count in (
        stats[
            "source_label_balance"
        ].items()
    ):

        print(
            f"{label}: "
            f"{count}"
        )

    print(
        "\nValid-sample label balance"
    )

    print("-" * 60)

    for label, count in (
        stats[
            "sample_label_balance"
        ].items()
    ):

        print(
            f"{label}: "
            f"{count}"
        )

    print(
        "\nQuiet-sample label balance"
    )

    print("-" * 60)

    for label, count in (
        stats[
            "quiet_sample_balance"
        ].items()
    ):

        print(
            f"{label}: "
            f"{count}"
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

    print(
        "\nFirst sample"
    )

    print("-" * 60)

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
        f"First label tensor: "
        f"{first_label}"
    )

    print(
        f"First label dtype: "
        f"{first_label.dtype}"
    )

    print(
        f"First label name: "
        f"{dataset.index_to_label[first_label.item()]}"
    )

    print(
        f"Number of patches: "
        f"{first_sample.shape[0]}"
    )

    print(
        f"Values per patch: "
        f"{first_sample.shape[1]}"
    )


# Check one DataLoader batch
def check_data_loader(
    data_loader,
    loader_name
):
    """
    Prints the shape and dtype of one batch.
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

    print("=" * 50)

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
        f"Label batch: "
        f"{label_batch}"
    )


# Testing
if __name__ == "__main__":

    # Create datasets
    (
        supervised_training_dataset,
        supervised_validation_dataset,
        unsupervised_training_dataset,
        unsupervised_validation_dataset,
        test_dataset,
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
            convert_to_cqt_spectrogram
        ),
        set_limit=(
            SET_TRACK_LIMIT
        ),
        maximum_tracks=(
            MAXIMUM_TRACKS
        )
    )

    # Retrieve model label mappings
    (
        model_label_to_index,
        model_index_to_label
    ) = (
        supervised_training_dataset
        .return_label_mappings()
    )

    # Check label mappings
    print(
        "\nLabel Mapping Comparison"
    )

    print("=" * 80)

    print(
        "Model / Training mapping:"
    )

    for label, index in (
        model_label_to_index.items()
    ):

        print(
            f"{index}: "
            f"{label}"
        )

    print(
        "\nModel mapping matches "
        "validation mapping:",
        model_label_to_index
        == label_to_index_val
    )

    print(
        "Model mapping matches "
        "test mapping:",
        model_label_to_index
        == label_to_index_test
    )

    print(
        f"\nNumber of model classes: "
        f"{len(model_label_to_index)}"
    )

    print(
        f"Number of validation classes: "
        f"{len(label_to_index_val)}"
    )

    print(
        f"Number of test classes: "
        f"{len(label_to_index_test)}"
    )

    # Dataset collection
    datasets = {
        "Sup Train":
            supervised_training_dataset,

        "Sup Val":
            supervised_validation_dataset,

        "Unsup Train":
            unsupervised_training_dataset,

        "Unsup Val":
            unsupervised_validation_dataset,

        "Test":
            test_dataset
    }

    # Compare general dataset statistics
    print(
        "\nDataset Statistics Comparison"
    )

    print("=" * 105)

    print(
        f"{'Statistic':<25}"
        f"{'Sup Train':>16}"
        f"{'Sup Val':>16}"
        f"{'Unsup Train':>16}"
        f"{'Unsup Val':>16}"
        f"{'Test':>16}"
    )

    print("-" * 105)

    statistic_keys = [
        "source_files",
        "valid_source_files",
        "invalid_source_files",
        "total_sub_samples",
        "valid_sub_samples",
        "quiet_sub_samples"
    ]

    for statistic in (
        statistic_keys
    ):

        print(
            f"{statistic:<25}",
            end=""
        )

        for dataset in (
            datasets.values()
        ):

            stats = (
                dataset
                .return_processing_stats()
            )

            print(
                f"{stats[statistic]:>16}",
                end=""
            )

        print()

    print(
        f"{'dataset_samples':<25}",
        end=""
    )

    for dataset in (
        datasets.values()
    ):

        print(
            f"{len(dataset):>16}",
            end=""
        )

    print()

    # Compare source label balance
    print(
        "\nSource File Label Balance"
    )

    print("=" * 105)

    print(
        f"{'Label':<25}"
        f"{'Sup Train':>16}"
        f"{'Sup Val':>16}"
        f"{'Unsup Train':>16}"
        f"{'Unsup Val':>16}"
        f"{'Test':>16}"
    )

    print("-" * 105)

    for label in (
        model_label_to_index
    ):

        print(
            f"{label:<25}",
            end=""
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
                    "source_label_balance"
                ].get(
                    label,
                    0
                )
            )

            print(
                f"{count:>16}",
                end=""
            )

        print()

    # Compare valid sample label balance
    print(
        "\nValid Sample Label Balance"
    )

    print("=" * 105)

    print(
        f"{'Label':<25}"
        f"{'Sup Train':>16}"
        f"{'Sup Val':>16}"
        f"{'Unsup Train':>16}"
        f"{'Unsup Val':>16}"
        f"{'Test':>16}"
    )

    print("-" * 105)

    for label in (
        model_label_to_index
    ):

        print(
            f"{label:<25}",
            end=""
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

            print(
                f"{count:>16}",
                end=""
            )

        print()

    # Compare quiet sample balance
    print(
        "\nQuiet Samples Removed Per Label"
    )

    print("=" * 105)

    print(
        f"{'Label':<25}"
        f"{'Sup Train':>16}"
        f"{'Sup Val':>16}"
        f"{'Unsup Train':>16}"
        f"{'Unsup Val':>16}"
        f"{'Test':>16}"
    )

    print("-" * 105)

    for label in (
        model_label_to_index
    ):

        print(
            f"{label:<25}",
            end=""
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
                    "quiet_sample_balance"
                ].get(
                    label,
                    0
                )
            )

            print(
                f"{count:>16}",
                end=""
            )

        print()

    # Check individual datasets
    check_supervised_dataset(
        supervised_training_dataset,
        "Supervised Training Dataset"
    )

    check_supervised_dataset(
        supervised_validation_dataset,
        "Supervised Validation Dataset"
    )

    check_supervised_dataset(
        unsupervised_training_dataset,
        "Unsupervised Training Dataset"
    )

    check_supervised_dataset(
        unsupervised_validation_dataset,
        "Unsupervised Validation Dataset"
    )

    check_supervised_dataset(
        test_dataset,
        "Test Dataset"
    )

    # Create DataLoaders
    supervised_training_loader = (
        DataLoader(
            supervised_training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=(
                SHUFFLE_TRAINING_DATA
            )
        )
    )

    supervised_validation_loader = (
        DataLoader(
            supervised_validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    unsupervised_training_loader = (
        DataLoader(
            unsupervised_training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=(
                SHUFFLE_TRAINING_DATA
            )
        )
    )

    unsupervised_validation_loader = (
        DataLoader(
            unsupervised_validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    # Check one batch from each DataLoader
    check_data_loader(
        supervised_training_loader,
        "Supervised Training Batch"
    )

    check_data_loader(
        supervised_validation_loader,
        "Supervised Validation Batch"
    )

    check_data_loader(
        unsupervised_training_loader,
        "Unsupervised Training Batch"
    )

    check_data_loader(
        unsupervised_validation_loader,
        "Unsupervised Validation Batch"
    )

    check_data_loader(
        test_loader,
        "Test Batch"
    )