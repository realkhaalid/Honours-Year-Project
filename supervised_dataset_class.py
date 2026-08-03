import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from data_processing_pipeline import (
    find_audio_files,
    load_and_validate_audio,
    cut_audio_to_consistent_length,
    split_audio_into_subsamples
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
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 16
SHUFFLE_TRAINING_DATA = True

SET_TRACK_LIMIT = False
MAXIMUM_TRACKS = 20

# Supervised BabySlakh dataset path
BABY_SLAKH_DATASET_PATH = Path(
    "C:/Users/ktyer/Downloads/"
    "Supervised-20260729T121921Z-1-001/"
    "Supervised/babyslakh_16k/babyslakh_16k"
)

# Supervised dataset class
class SupervisedAudioDataset(Dataset):
    """
    Stores valid two-second audio samples and their
    corresponding instrument labels.
    """

    def __init__(
        self,
        labelled_audio_files,
        label_to_index
    ):
        self.samples = []
        self.labels = []

        self.label_to_index = label_to_index

        self.index_to_label = {
            index: label
            for label, index in label_to_index.items()
        }

        self.processing_stats = {
            "source_files": len(labelled_audio_files),
            "valid_source_files": 0,
            "invalid_source_files": 0,
            "total_sub_samples": 0,
            "valid_sub_samples": 0,
            "quiet_sub_samples": 0
        }

        self._process_audio_files(
            labelled_audio_files
        )

    def _process_audio_files(
        self,
        labelled_audio_files
    ):
        """
        Loads and processes every labelled audio stem.
        """

        for file_number, (
            audio_file,
            instrument_label
        ) in enumerate(
            labelled_audio_files,
            start=1
        ):
            print(
                f"[{file_number}/"
                f"{len(labelled_audio_files)}] "
                f"Processing: {audio_file.name} | "
                f"Label: {instrument_label}"
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

                valid_patches, statistics = (
                    split_audio_into_subsamples(
                        audio_cut,
                        sample_rate
                    )
                )

                self._store_valid_samples(
                    valid_patches,
                    instrument_label
                )

                self._update_processing_stats(
                    statistics
                )

                self.processing_stats[
                    "valid_source_files"
                ] += 1

                print(
                    f"  Valid samples: "
                    f"{statistics['valid_sub_samples']}"
                )

                print(
                    f"  Quiet samples removed: "
                    f"{statistics['quiet_sub_samples']}"
                )

            except (
                ValueError,
                TypeError,
                KeyError
            ) as error:
                self.processing_stats[
                    "invalid_source_files"
                ] += 1

                print(
                    f"  Invalid file: "
                    f"{audio_file.name}"
                )

                print(
                    f"  Reason: {error}"
                )

    def return_label_mappings(self):
        """
        Returns the label-to-index and index-to-label
        mappings.
        """

        return (
            self.label_to_index,
            self.index_to_label
        )            

    def _store_valid_samples(
        self,
        valid_patches,
        instrument_label
    ):
        """
        Stores each valid patch tensor and its label.

        Input patch shape:

            [1, number_of_patches, patch_features]

        Stored patch shape:

            [number_of_patches, patch_features]
        """

        if instrument_label not in self.label_to_index:
            raise KeyError(
                f"Unknown instrument label: "
                f"{instrument_label}"
            )

        label_index = self.label_to_index[
            instrument_label
        ]

        for patch_tensor in valid_patches:
            if not isinstance(
                patch_tensor,
                torch.Tensor
            ):
                raise TypeError(
                    "Each processed sample must be a "
                    "PyTorch tensor."
                )

            if patch_tensor.ndim != 3:
                raise ValueError(
                    "Expected patch tensor shape "
                    "[1, number_of_patches, patch_features], "
                    f"but received {patch_tensor.shape}."
                )

            if patch_tensor.shape[0] != 1:
                raise ValueError(
                    "The first patch tensor dimension must "
                    "be 1 before storing the sample."
                )

            sample = patch_tensor.squeeze(0)

            self.samples.append(
                sample.to(dtype=torch.float32)
            )

            self.labels.append(
                label_index
            )

    def _update_processing_stats(
        self,
        statistics
    ):
        """
        Adds one source file's processing statistics to the
        overall dataset statistics.
        """

        self.processing_stats[
            "total_sub_samples"
        ] += statistics["total_sub_samples"]

        self.processing_stats[
            "valid_sub_samples"
        ] += statistics["valid_sub_samples"]

        self.processing_stats[
            "quiet_sub_samples"
        ] += statistics["quiet_sub_samples"]

    def __len__(self):
        """
        Returns the number of valid labelled samples.
        """

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):
        """
        Returns one processed sample and its integer label.
        """

        sample = self.samples[index]

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return sample, label


# ---------------------------------------------------------
# Collect labelled BabySlakh files
# ---------------------------------------------------------

def collect_supervised_audio_files(
    dataset_path,
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS
):
    """
    Finds rendered BabySlakh stems and retrieves their
    instrument labels from metadata.yaml.

    find_audio_files() should return a list containing:

        (audio_file_path, instrument_label)
    """

    labelled_audio_files = find_audio_files(
        dataset_path=dataset_path,
        set_limit=set_limit,
        maximum_files=maximum_tracks,
        supervised=True
    )

    print(
        f"\nTotal labelled source files found: "
        f"{len(labelled_audio_files)}"
    )

    return labelled_audio_files


# ---------------------------------------------------------
# Create label mapping
# ---------------------------------------------------------

def create_label_mapping(
    labelled_audio_files
):
    """
    Creates an integer index for each unique instrument
    label.
    """

    unique_labels = sorted({
        label
        for _, label in labelled_audio_files
    })

    if not unique_labels:
        raise ValueError(
            "No instrument labels were found."
        )

    label_to_index = {
        label: index
        for index, label in enumerate(
            unique_labels
        )
    }

    return label_to_index


# ---------------------------------------------------------
# Split labelled source files
# ---------------------------------------------------------

def split_labelled_audio_files(
    labelled_audio_files,
    validation_ratio=VALIDATION_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED
):
    """
    Splits labelled source files into training,
    validation, and test groups.

    The source files are split before they are divided into
    two-second samples.
    """

    if len(labelled_audio_files) < 3:
        raise ValueError(
            "At least three labelled audio files are "
            "required."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    if validation_ratio + test_ratio >= 1.0:
        raise ValueError(
            "validation_ratio and test_ratio must add up "
            "to less than 1."
        )

    shuffled_files = list(
        labelled_audio_files
    )

    random_generator = random.Random(
        random_seed
    )

    random_generator.shuffle(
        shuffled_files
    )

    total_files = len(
        shuffled_files
    )

    validation_size = max(
        1,
        int(
            total_files
            * validation_ratio
        )
    )

    test_size = max(
        1,
        int(
            total_files
            * test_ratio
        )
    )

    training_size = (
        total_files
        - validation_size
        - test_size
    )

    if training_size < 1:
        raise ValueError(
            "The training split contains no files."
        )

    training_files = shuffled_files[
        :training_size
    ]

    validation_files = shuffled_files[
        training_size:
        training_size + validation_size
    ]

    test_files = shuffled_files[
        training_size + validation_size:
    ]

    return (
        training_files,
        validation_files,
        test_files
    )


# ---------------------------------------------------------
# Create supervised datasets
# ---------------------------------------------------------

def create_supervised_datasets(
    dataset_path,
    validation_ratio=VALIDATION_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED,
    set_limit=SET_TRACK_LIMIT,
    maximum_tracks=MAXIMUM_TRACKS
):
    """
    Collects BabySlakh stems, creates a shared label
    mapping, performs the train-validation-test split, and
    returns the three supervised datasets.
    """

    labelled_audio_files = (
        collect_supervised_audio_files(
            dataset_path,
            set_limit=set_limit,
            maximum_tracks=maximum_tracks
        )
    )

    if not labelled_audio_files:
        raise ValueError(
            "No labelled audio files were found."
        )

    label_to_index = create_label_mapping(
        labelled_audio_files
    )

    (
        training_files,
        validation_files,
        test_files
    ) = split_labelled_audio_files(
        labelled_audio_files,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    print("\nLabel mapping")
    print("-" * 50)

    for label, index in label_to_index.items():
        print(f"{index}: {label}")

    print("\nSource-file split")
    print("-" * 50)

    print(
        f"Training source files: "
        f"{len(training_files)}"
    )

    print(
        f"Validation source files: "
        f"{len(validation_files)}"
    )

    print(
        f"Test source files: "
        f"{len(test_files)}"
    )

    print("\nCreating supervised training dataset")
    print("-" * 50)

    training_dataset = SupervisedAudioDataset(
        training_files,
        label_to_index
    )

    _, index_to_label = training_dataset.return_label_mappings()

    print("\nCreating supervised validation dataset")
    print("-" * 50)

    validation_dataset = SupervisedAudioDataset(
        validation_files,
        label_to_index
    )

    print("\nCreating supervised test dataset")
    print("-" * 50)

    test_dataset = SupervisedAudioDataset(
        test_files,
        label_to_index
    )

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
        label_to_index,
        index_to_label
    )


# ---------------------------------------------------------
# Check dataset information
# ---------------------------------------------------------

def check_supervised_dataset(
    dataset,
    dataset_name
):
    """
    Prints information about a supervised dataset.
    """

    print(f"\n{dataset_name}")
    print("=" * 50)

    print(
        f"Number of samples: "
        f"{len(dataset)}"
    )

    print(
        f"Processing statistics: "
        f"{dataset.processing_stats}"
    )

    print(
        f"Number of classes: "
        f"{len(dataset.label_to_index)}"
    )

    if len(dataset) == 0:
        print(
            "The dataset contains no valid samples."
        )
        return

    first_sample, first_label = dataset[0]

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


# ---------------------------------------------------------
# Check one DataLoader batch
# ---------------------------------------------------------

def check_data_loader(
    data_loader,
    loader_name
):
    """
    Prints the shape and dtype of one batch.
    """

    if len(data_loader.dataset) == 0:
        print(
            f"\n{loader_name} contains no samples."
        )
        return

    sample_batch, label_batch = next(
        iter(data_loader)
    )

    print(f"\n{loader_name}")
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    (
        training_dataset,
        validation_dataset,
        test_dataset,
        label_to_index
    ) = create_supervised_datasets(
        dataset_path=BABY_SLAKH_DATASET_PATH,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
        set_limit=SET_TRACK_LIMIT,
        maximum_tracks=MAXIMUM_TRACKS
    )

    check_supervised_dataset(
        training_dataset,
        "Supervised training dataset"
    )

    check_supervised_dataset(
        validation_dataset,
        "Supervised validation dataset"
    )

    check_supervised_dataset(
        test_dataset,
        "Supervised test dataset"
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAINING_DATA
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    check_data_loader(
        training_loader,
        "Supervised training batch"
    )

    check_data_loader(
        validation_loader,
        "Supervised validation batch"
    )

    check_data_loader(
        test_loader,
        "Supervised test batch"
    )