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

#ignore librosa warnings
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

#Config
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42

BATCH_SIZE = 16
SHUFFLE_TRAINING_DATA = True

#Unsupervised Datasets
QUARTET_PATH = Path("C:/Users/ktyer/Downloads/Quartet-20260729T123130Z-1-001/Quartet")
PIANO_PATH = Path("C:/Users/ktyer/Downloads/Piano solo 1-20260729T130121Z-1-001/Piano solo 1")
ACAPELLA_PATH = Path("C:/Users/ktyer/Downloads/Acappella-20260729T145009Z-1-001/Acappella")
RUMBACHONTA_PATH = Path("C:/Users/ktyer/Downloads/AlejoGranados_RumbaChonta_Full-20260729T150527Z-1-001/AlejoGranados_RumbaChonta_Full")
DEADROSES_PATH = Path("C:/Users/ktyer/Downloads/AndrewCole_DeadRoses_Full-20260729T150533Z-1-001/AndrewCole_DeadRoses_Full")
CORINE_PATH = Path("C:/Users/ktyer/Downloads/AbletonesBigBand_CorineCorine_Full-20260729T150514Z-1-001/AbletonesBigBand_CorineCorine_Full")
SONGOFINDIA_PATH = Path("C:/Users/ktyer/Downloads/AbletonesBigBand_SongOfIndia_Full-20260729T150519Z-1-001/AbletonesBigBand_SongOfIndia_Full")

UNSUPERVISED_DATASET_PATHS = [
    QUARTET_PATH,
    PIANO_PATH,
    ACAPELLA_PATH,
    RUMBACHONTA_PATH,
    DEADROSES_PATH,
    CORINE_PATH,
    SONGOFINDIA_PATH
]

class UnsupervisedAudioDataset(Dataset):
    """
    Stores valid two-second audio samples after they have
    been converted into log-mel spectrogram patches.
    """

    def __init__(self, audio_files):
        self.samples: list[torch.Tensor] = []

        self.processing_stats = {
            "source_files": len(audio_files),
            "valid_source_files": 0,
            "invalid_source_files": 0,
            "total_sub_samples": 0,
            "valid_sub_samples": 0,
            "quiet_sub_samples": 0
        }

        self._process_audio_files(audio_files)

    def _process_audio_files(
        self,
        audio_files
    ):
        """
        Loads and processes every source audio file.
        """

        for file_number, audio_file in enumerate(
            audio_files,
            start=1
        ):
            print(
                f"[{file_number}/{len(audio_files)}] "
                f"Processing: {audio_file.name}"
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
                    valid_patches
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

            except ValueError as error:
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

    def _store_valid_samples(
        self,
        valid_patches
    ):
        """
        Removes the temporary batch dimension and stores
        each valid sample.
        """

        for patch_tensor in valid_patches:
            if not isinstance(
                patch_tensor,
                torch.Tensor
            ):
                raise TypeError(
                    "Each processed sample must be a PyTorch tensor."
                )

            if patch_tensor.ndim != 3:
                raise ValueError(
                    "Expected patch tensor shape [1, number_of_patches, patch_features],"
                    f"but received {patch_tensor.shape}."
                )

            if patch_tensor.shape[0] != 1:
                raise ValueError(
                    "The first patch tensor dimension must be 1 before storing the sample."
                )

            sample = patch_tensor.squeeze(0)

            self.samples.append(
                sample.to(dtype=torch.float32)
            )

    def _update_processing_stats(
        self,
        statistics
    ):
        """
        Adds one file's processing statistics to the
        complete dataset statistics.
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
        Returns the number of valid two-second samples.
        """

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):
        """
        Returns one processed sample.
        """

        return self.samples[index]

# Find audio files
def collect_unsupervised_audio_files(
    dataset_paths,
    set_limit=False,
    maximum_files=20
):
    """
    Finds audio files from every unsupervised dataset
    folder.
    """
    
    all_audio_files = []

    for dataset_path in dataset_paths:

        if dataset_path in {
            QUARTET_PATH,
            PIANO_PATH,
            ACAPELLA_PATH
        }:
            set_limit = True
        else:
            set_limit = False

        audio_files = find_audio_files(
            dataset_path,
            set_limit=set_limit,
            maximum_files=maximum_files
        )

        all_audio_files.extend(
            audio_files
        )

        print(
            f"{dataset_path.name}: "
            f"{len(audio_files)} files found"
        )

    print(
        f"\nTotal source files found: "
        f"{len(all_audio_files)}"
    )

    return all_audio_files

# Split audio files
def split_audio_files(
    audio_files,
    validation_ratio=VALIDATION_RATIO,
    random_seed=RANDOM_SEED
):
    """
    Splits source audio files into training and validation
    groups.

    The split happens before creating two-second samples,
    which helps prevent samples from the same source file
    appearing in both datasets.
    """

    if len(audio_files) < 2:
        raise ValueError(
            "At least two audio files are required."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    shuffled_files = list(audio_files)

    random_generator = random.Random(
        random_seed
    )

    random_generator.shuffle(
        shuffled_files
    )

    validation_size = max(
        1,
        int(
            len(shuffled_files)
            * validation_ratio
        )
    )

    validation_files = shuffled_files[
        :validation_size
    ]

    training_files = shuffled_files[
        validation_size:
    ]

    if len(training_files) == 0:
        raise ValueError(
            "The training split contains no files."
        )

    return training_files, validation_files

# Create unsupervised datasets
def create_unsupervised_datasets(
    dataset_paths,
    validation_ratio=VALIDATION_RATIO,
    random_seed=RANDOM_SEED
):
    """
    Finds source files, splits them and creates the training
    and validation datasets.
    """

    all_audio_files = (
        collect_unsupervised_audio_files(
            dataset_paths
        )
    )

    training_files, validation_files = (
        split_audio_files(
            all_audio_files,
            validation_ratio=validation_ratio,
            random_seed=random_seed
        )
    )

    print("\nSource-file split")
    print(
        f"Training source files: "
        f"{len(training_files)}"
    )
    print(
        f"Validation source files: "
        f"{len(validation_files)}"
    )

    print("\nCreating training dataset")
    print("-" * 50)

    training_dataset = UnsupervisedAudioDataset(
        training_files
    )

    print("\nCreating validation dataset")
    print("-" * 50)

    validation_dataset = UnsupervisedAudioDataset(
        validation_files
    )

    return training_dataset, validation_dataset

# Check dataset info
def check_dataset(
    dataset,
    dataset_name
):
    """
    Prints basic information about a created dataset.
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

    if len(dataset) == 0:
        print(
            "The dataset contains no valid samples."
        )
        return

    first_sample = dataset[0]

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
        f"Number of patches: "
        f"{first_sample.shape[0]}"
    )

    print(
        f"Values per patch: "
        f"{first_sample.shape[1]}"
    )

if __name__ == "__main__":
    training_dataset, validation_dataset = (
        create_unsupervised_datasets(
            UNSUPERVISED_DATASET_PATHS,
            validation_ratio=VALIDATION_RATIO,
            random_seed=RANDOM_SEED
        )
    )

    check_dataset(
        training_dataset,
        "Training dataset"
    )

    check_dataset(
        validation_dataset,
        "Validation dataset"
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

    if len(training_dataset) > 0:
        training_batch = next(
            iter(training_loader)
        )

        print("\nTraining batch")
        print("=" * 50)

        print(
            f"Batch shape: "
            f"{training_batch.shape}"
        )

        print(
            f"Batch dtype: "
            f"{training_batch.dtype}"
        )

    if len(validation_dataset) > 0:
        validation_batch = next(
            iter(validation_loader)
        )

        print("\nValidation batch")
        print("=" * 50)

        print(
            f"Batch shape: "
            f"{validation_batch.shape}"
        )

        print(
            f"Batch dtype: "
            f"{validation_batch.dtype}"
        )