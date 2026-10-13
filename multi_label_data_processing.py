import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_processing_pipeline import (
    LOUDNESS_THRESHOLD_DB,
    TARGET_SAMPLE_RATE,
    SOURCE_DURATION_SECONDS,
    CLIP_DURATION_SECONDS,
    SUPPORTED_EXTENSIONS,
    EXCLUDED_LABELS,
    load_and_validate_audio,
    cut_audio_to_consistent_length,
    calculate_loudness_db,
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


# ---------------------------------------------------------------------
# Multi-label configuration
# ---------------------------------------------------------------------

# Keep the same class order used by the current single-label experiment.
MULTI_LABEL_MAPPING = {
    "Bass": 0,
    "Brass": 1,
    "Chromatic Percussion": 2,
    "Drums": 3,
    "Guitar": 4,
    "Organ": 5,
    "Piano": 6,
    "Pipe": 7,
    "Reed": 8,
    "Strings": 9,
    "Strings (continued)": 10,
    "Synth Lead": 11,
    "Synth Pad": 12
}

INDEX_TO_MULTI_LABEL = {
    index: label
    for label, index
    in MULTI_LABEL_MAPPING.items()
}

NUM_CLASSES = len(
    MULTI_LABEL_MAPPING
)

# The same threshold used to discard quiet single-label samples is used
# to decide whether a stem is active in a 5-second mixture sample.
STEM_ACTIVE_LOUDNESS_THRESHOLD_DB = (
    LOUDNESS_THRESHOLD_DB
)

# The mixture itself can also be filtered using the same threshold.
MIX_LOUDNESS_THRESHOLD_DB = (
    LOUDNESS_THRESHOLD_DB
)

MIX_FILE_NAME = "mix.flac"
METADATA_FILE_NAME = "metadata.yaml"
STEMS_FOLDER_NAME = "stems"


# ---------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------

def labels_to_multi_hot(
    labels,
    label_to_index=MULTI_LABEL_MAPPING
):
    """
    Converts a collection of instrument labels into
    a multi-hot float tensor.

    Example:
        ["Bass", "Drums", "Guitar"]

    becomes:
        [1, 0, 0, 1, 1, ...]
    """

    label_vector = torch.zeros(
        len(label_to_index),
        dtype=torch.float32
    )

    for label in labels:

        if label not in label_to_index:
            raise KeyError(
                f"Unknown instrument label: "
                f"{label}"
            )

        label_index = (
            label_to_index[
                label
            ]
        )

        label_vector[
            label_index
        ] = 1.0

    return label_vector


def multi_hot_to_labels(
    label_vector,
    index_to_label=INDEX_TO_MULTI_LABEL
):
    """
    Converts a multi-hot label tensor back into
    instrument label names.
    """
    

    if not isinstance(
        label_vector,
        torch.Tensor
    ):
        label_vector = torch.tensor(
            label_vector,
            dtype=torch.float32
        )

    if label_vector.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional "
            "multi-hot label vector."
        )

    labels = []

    for index, value in enumerate(
        label_vector.tolist()
    ):

        if value >= 0.5:

            if index not in index_to_label:
                raise KeyError(
                    f"Unknown label index: "
                    f"{index}"
                )

            labels.append(
                index_to_label[index]
            )

    return labels


# ---------------------------------------------------------------------
# Track and metadata helpers
# ---------------------------------------------------------------------

def find_track_folders(
    dataset_path,
    set_limit=False,
    maximum_tracks=20
):
    """
    Finds Slakh track folders directly inside a
    train, validation, or test directory.
    """

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: "
            f"{dataset_path}"
        )

    track_folders = sorted([
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    ])

    if set_limit:
        track_folders = (
            track_folders[
                :maximum_tracks
            ]
        )

    return track_folders


def load_track_metadata(
    track_folder
):
    """
    Loads metadata.yaml for one Slakh track.
    """

    track_folder = Path(
        track_folder
    )

    metadata_path = (
        track_folder
        / METADATA_FILE_NAME
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file missing: "
            f"{metadata_path}"
        )

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as metadata_file:

            metadata = yaml.safe_load(
                metadata_file
            )

    except (
        OSError,
        yaml.YAMLError
    ) as error:

        raise ValueError(
            f"Could not read metadata for "
            f"{track_folder.name}: "
            f"{error}"
        ) from error

    if not isinstance(
        metadata,
        dict
    ):
        raise ValueError(
            f"Invalid metadata format for "
            f"{track_folder.name}."
        )

    return metadata


def get_track_stem_information(
    track_folder,
    label_to_index=MULTI_LABEL_MAPPING
):
    """
    Returns every usable stem path and its
    instrument class for one track.
    """

    track_folder = Path(
        track_folder
    )

    stems_folder = (
        track_folder
        / STEMS_FOLDER_NAME
    )

    if not stems_folder.exists():
        raise FileNotFoundError(
            f"Stems folder missing: "
            f"{stems_folder}"
        )

    metadata = load_track_metadata(
        track_folder
    )

    stems_metadata = (
        metadata.get(
            "stems",
            {}
        )
    )

    if not isinstance(
        stems_metadata,
        dict
    ):
        raise ValueError(
            f"Invalid stems metadata for "
            f"{track_folder.name}."
        )

    stem_information = []

    stem_files = sorted([
        stem_file
        for stem_file in stems_folder.iterdir()
        if (
            stem_file.is_file()
            and stem_file.suffix.lower()
            in SUPPORTED_EXTENSIONS
            and not stem_file.name.startswith(
                "._"
            )
        )
    ])

    for stem_path in stem_files:

        stem_id = (
            stem_path.stem
        )

        information = (
            stems_metadata.get(
                stem_id
            )
        )

        if information is None:
            continue

        instrument_label = (
            information.get(
                "inst_class"
            )
        )

        if instrument_label is None:
            continue

        if instrument_label in EXCLUDED_LABELS:
            continue

        if instrument_label not in label_to_index:
            continue

        stem_information.append(
            {
                "stem_path":
                    stem_path,

                "stem_id":
                    stem_id,

                "label":
                    instrument_label,

                "label_index":
                    label_to_index[
                        instrument_label
                    ]
            }
        )

    if len(stem_information) == 0:
        raise ValueError(
            f"No usable labelled stems found "
            f"for track {track_folder.name}."
        )

    return stem_information


# ---------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------

def get_mix_path(
    track_folder
):
    """
    Returns the mix.flac path for one track.
    """

    mix_path = (
        Path(track_folder)
        / MIX_FILE_NAME
    )

    if not mix_path.exists():
        raise FileNotFoundError(
            f"Mix file missing: "
            f"{mix_path}"
        )

    return mix_path


def load_track_mix(
    track_folder
):
    """
    Loads and validates the track mixture, then
    applies the same source-length rule used by the
    single-label processing pipeline.
    """

    mix_path = get_mix_path(
        track_folder
    )

    mix_audio, sample_rate = (
        load_and_validate_audio(
            mix_path
        )
    )

    mix_audio = (
        cut_audio_to_consistent_length(
            mix_audio,
            sample_rate
        )
    )

    return (
        mix_path,
        mix_audio,
        sample_rate
    )


def load_track_stems(
    track_folder,
    target_length,
    sample_rate=TARGET_SAMPLE_RATE,
    label_to_index=MULTI_LABEL_MAPPING
):
    """
    Loads all usable stems for one track.

    Each loaded stem is aligned to the same starting
    position as mix.flac and clipped to target_length.
    """

    stem_information = (
        get_track_stem_information(
            track_folder,
            label_to_index=(
                label_to_index
            )
        )
    )

    loaded_stems = []
    invalid_stems = []

    for information in stem_information:

        stem_path = (
            information[
                "stem_path"
            ]
        )

        try:

            stem_audio, stem_sample_rate = (
                load_and_validate_audio(
                    stem_path
                )
            )

            if (
                stem_sample_rate
                != sample_rate
            ):
                raise ValueError(
                    "Stem sample rate does not "
                    "match mixture sample rate."
                )

            if len(stem_audio) < target_length:
                raise ValueError(
                    f"Stem is shorter than the "
                    f"processed mixture. "
                    f"Stem samples: "
                    f"{len(stem_audio)}, "
                    f"required: "
                    f"{target_length}."
                )

            stem_audio = (
                stem_audio[
                    :target_length
                ]
            )

            loaded_stems.append(
                {
                    **information,
                    "audio":
                        stem_audio
                }
            )

        except (
            ValueError,
            TypeError,
            OSError
        ) as error:

            invalid_stems.append(
                {
                    "stem_path":
                        str(
                            stem_path
                        ),

                    "label":
                        information[
                            "label"
                        ],

                    "reason":
                        str(
                            error
                        )
                }
            )

    if len(loaded_stems) == 0:
        raise ValueError(
            f"No valid stems could be loaded "
            f"for track "
            f"{Path(track_folder).name}."
        )

    return (
        loaded_stems,
        invalid_stems
    )


# ---------------------------------------------------------------------
# Activity detection
# ---------------------------------------------------------------------

def find_active_stems_for_interval(
    loaded_stems,
    start_sample,
    end_sample,
    loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    )
):
    """
    Determines which stems are active inside one
    aligned 5-second interval.

    A stem is active when its RMS loudness is at or
    above the configured dBFS threshold.
    """

    expected_length = (
        end_sample
        - start_sample
    )

    active_labels = set()
    active_stems = []

    for stem in loaded_stems:

        stem_subsample = (
            stem["audio"][
                start_sample:end_sample
            ]
        )

        if (
            len(stem_subsample)
            != expected_length
        ):
            continue

        stem_loudness_db = (
            calculate_loudness_db(
                stem_subsample
            )
        )

        if (
            stem_loudness_db
            < loudness_threshold
        ):
            continue

        active_labels.add(
            stem["label"]
        )

        active_stems.append(
            {
                "stem_path":
                    stem[
                        "stem_path"
                    ],

                "stem_id":
                    stem[
                        "stem_id"
                    ],

                "label":
                    stem[
                        "label"
                    ],

                "label_index":
                    stem[
                        "label_index"
                    ],

                "loudness_db":
                    float(
                        stem_loudness_db
                    )
            }
        )

    active_labels = sorted(
        active_labels,
        key=lambda label: (
            MULTI_LABEL_MAPPING[
                label
            ]
        )
    )

    return (
        active_labels,
        active_stems
    )


def create_multilabel_sample_metadata(
    track_folder,
    mix_path,
    sample_rate,
    start_sample,
    end_sample,
    mix_loudness_db,
    active_labels,
    active_stems,
    label_to_index=MULTI_LABEL_MAPPING
):
    """
    Creates one lightweight metadata record for a
    valid multi-label mixture sample.
    """

    label_vector = (
        labels_to_multi_hot(
            active_labels,
            label_to_index=(
                label_to_index
            )
        )
    )

    label_indices = [
        label_to_index[
            label
        ]
        for label in active_labels
    ]

    sample = {
        "track":
            Path(
                track_folder
            ).name,

        "track_folder":
            Path(
                track_folder
            ),

        "mix_path":
            Path(
                mix_path
            ),

        "start_sample":
            int(
                start_sample
            ),

        "end_sample":
            int(
                end_sample
            ),

        "start_time":
            float(
                start_sample
                / sample_rate
            ),

        "end_time":
            float(
                end_sample
                / sample_rate
            ),

        "sample_rate":
            int(
                sample_rate
            ),

        "mix_loudness_db":
            float(
                mix_loudness_db
            ),

        "labels":
            tuple(
                active_labels
            ),

        "label_indices":
            tuple(
                label_indices
            ),

        "label_vector":
            label_vector,

        "active_stems":
            active_stems
    }

    return sample


# ---------------------------------------------------------------------
# Track processing
# ---------------------------------------------------------------------

def process_multilabel_track(
    track_folder,
    clip_duration=CLIP_DURATION_SECONDS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    label_to_index=MULTI_LABEL_MAPPING
):
    """
    Processes one complete Slakh track.

    mix.flac defines each 5-second input sample.
    Every stem is checked over the exact same time
    interval to determine the multi-label target.

    Returns lightweight sample metadata only.
    Spectrograms and patches are not permanently
    stored here.
    """

    track_folder = Path(
        track_folder
    )

    (
        mix_path,
        mix_audio,
        sample_rate
    ) = (
        load_track_mix(
            track_folder
        )
    )

    (
        loaded_stems,
        invalid_stems
    ) = (
        load_track_stems(
            track_folder=(
                track_folder
            ),
            target_length=(
                len(
                    mix_audio
                )
            ),
            sample_rate=(
                sample_rate
            ),
            label_to_index=(
                label_to_index
            )
        )
    )

    sub_sample_length = int(
        sample_rate
        * clip_duration
    )

    samples = []

    total_sub_samples = 0
    quiet_mix_sub_samples = 0
    no_active_label_sub_samples = 0

    label_sample_counts = {
        label: 0
        for label
        in label_to_index
    }

    label_combination_counts = {}

    for start_sample in range(
        0,
        len(mix_audio),
        sub_sample_length
    ):

        end_sample = (
            start_sample
            + sub_sample_length
        )

        mix_subsample = (
            mix_audio[
                start_sample:end_sample
            ]
        )

        # Ignore incomplete final interval.
        if (
            len(mix_subsample)
            != sub_sample_length
        ):
            continue

        total_sub_samples += 1

        mix_loudness_db = (
            calculate_loudness_db(
                mix_subsample
            )
        )

        if (
            mix_loudness_db
            < mix_loudness_threshold
        ):

            quiet_mix_sub_samples += 1
            continue

        (
            active_labels,
            active_stems
        ) = (
            find_active_stems_for_interval(
                loaded_stems=(
                    loaded_stems
                ),
                start_sample=(
                    start_sample
                ),
                end_sample=(
                    end_sample
                ),
                loudness_threshold=(
                    stem_loudness_threshold
                )
            )
        )

        if len(active_labels) == 0:

            no_active_label_sub_samples += 1
            continue

        sample = (
            create_multilabel_sample_metadata(
                track_folder=(
                    track_folder
                ),
                mix_path=(
                    mix_path
                ),
                sample_rate=(
                    sample_rate
                ),
                start_sample=(
                    start_sample
                ),
                end_sample=(
                    end_sample
                ),
                mix_loudness_db=(
                    mix_loudness_db
                ),
                active_labels=(
                    active_labels
                ),
                active_stems=(
                    active_stems
                ),
                label_to_index=(
                    label_to_index
                )
            )
        )

        samples.append(
            sample
        )

        for label in active_labels:

            label_sample_counts[
                label
            ] += 1

        combination_key = tuple(
            active_labels
        )

        label_combination_counts[
            combination_key
        ] = (
            label_combination_counts.get(
                combination_key,
                0
            )
            + 1
        )

    statistics = {
        "track":
            track_folder.name,

        "mix_path":
            str(
                mix_path
            ),

        "sample_rate":
            sample_rate,

        "processed_duration_seconds":
            len(
                mix_audio
            )
            / sample_rate,

        "source_stems":
            len(
                loaded_stems
            )
            + len(
                invalid_stems
            ),

        "valid_stems":
            len(
                loaded_stems
            ),

        "invalid_stems":
            invalid_stems,

        "total_sub_samples":
            total_sub_samples,

        "valid_sub_samples":
            len(
                samples
            ),

        "quiet_mix_sub_samples":
            quiet_mix_sub_samples,

        "no_active_label_sub_samples":
            no_active_label_sub_samples,

        "sample_label_balance":
            label_sample_counts,

        "label_combination_balance":
            label_combination_counts
    }

    # Release the largest temporary objects before
    # moving on to the next track.
    del loaded_stems
    del mix_audio

    return (
        samples,
        statistics
    )


# ---------------------------------------------------------------------
# Dataset split processing
# ---------------------------------------------------------------------

def build_multilabel_sample_index(
    dataset_path,
    set_limit=False,
    maximum_tracks=20,
    clip_duration=CLIP_DURATION_SECONDS,
    mix_loudness_threshold=(
        MIX_LOUDNESS_THRESHOLD_DB
    ),
    stem_loudness_threshold=(
        STEM_ACTIVE_LOUDNESS_THRESHOLD_DB
    ),
    label_to_index=MULTI_LABEL_MAPPING,
    show_progress=True
):
    """
    Processes every track in one Slakh split and
    builds a lightweight multi-label sample index.

    No spectrogram or patch tensors are permanently
    stored. Those can be produced lazily later.
    """

    track_folders = (
        find_track_folders(
            dataset_path=(
                dataset_path
            ),
            set_limit=(
                set_limit
            ),
            maximum_tracks=(
                maximum_tracks
            )
        )
    )

    samples = []

    overall_stats = {
        "tracks_found":
            len(
                track_folders
            ),

        "valid_tracks":
            0,

        "invalid_tracks":
            0,

        "total_sub_samples":
            0,

        "valid_sub_samples":
            0,

        "quiet_mix_sub_samples":
            0,

        "no_active_label_sub_samples":
            0,

        "sample_label_balance": {
            label: 0
            for label
            in label_to_index
        },

        "label_combination_balance":
            {},

        "invalid_tracks_details":
            [],

        "track_statistics":
            []
    }

    iterator = track_folders

    if show_progress:
        iterator = tqdm(
            track_folders,
            desc="Processing multi-label tracks"
        )

    for track_folder in iterator:

        try:

            (
                track_samples,
                track_stats
            ) = (
                process_multilabel_track(
                    track_folder=(
                        track_folder
                    ),
                    clip_duration=(
                        clip_duration
                    ),
                    mix_loudness_threshold=(
                        mix_loudness_threshold
                    ),
                    stem_loudness_threshold=(
                        stem_loudness_threshold
                    ),
                    label_to_index=(
                        label_to_index
                    )
                )
            )

            samples.extend(
                track_samples
            )

            overall_stats[
                "valid_tracks"
            ] += 1

            overall_stats[
                "total_sub_samples"
            ] += (
                track_stats[
                    "total_sub_samples"
                ]
            )

            overall_stats[
                "valid_sub_samples"
            ] += (
                track_stats[
                    "valid_sub_samples"
                ]
            )

            overall_stats[
                "quiet_mix_sub_samples"
            ] += (
                track_stats[
                    "quiet_mix_sub_samples"
                ]
            )

            overall_stats[
                "no_active_label_sub_samples"
            ] += (
                track_stats[
                    "no_active_label_sub_samples"
                ]
            )

            for label, count in (
                track_stats[
                    "sample_label_balance"
                ].items()
            ):

                overall_stats[
                    "sample_label_balance"
                ][label] += count

            for (
                label_combination,
                count
            ) in (
                track_stats[
                    "label_combination_balance"
                ].items()
            ):

                overall_stats[
                    "label_combination_balance"
                ][label_combination] = (
                    overall_stats[
                        "label_combination_balance"
                    ].get(
                        label_combination,
                        0
                    )
                    + count
                )

            overall_stats[
                "track_statistics"
            ].append(
                track_stats
            )

        except (
            FileNotFoundError,
            ValueError,
            TypeError,
            KeyError,
            OSError
        ) as error:

            overall_stats[
                "invalid_tracks"
            ] += 1

            overall_stats[
                "invalid_tracks_details"
            ].append(
                {
                    "track":
                        track_folder.name,

                    "reason":
                        str(
                            error
                        )
                }
            )

            if show_progress:
                tqdm.write(
                    f"Skipping "
                    f"{track_folder.name}: "
                    f"{error}"
                )

    return (
        samples,
        overall_stats
    )


# ---------------------------------------------------------------------
# Lazy mixture processing
# ---------------------------------------------------------------------

def load_multilabel_mix_sample(
    sample_information,
    data_representation=(
        convert_to_mel_spectrogram
    )
):
    """
    Lazily loads one indexed 5-second mix.flac sample
    and converts it into Transformer patches.

    Returns:
        patches:
            [number_of_patches, patch_dim]

        label_vector:
            [number_of_classes]
    """

    required_keys = (
        "mix_path",
        "start_sample",
        "sample_rate",
        "label_vector"
    )

    for key in required_keys:

        if key not in sample_information:
            raise KeyError(
                f"Sample metadata missing "
                f"required key: {key}"
            )

    (
        audio,
        sample_rate
    ) = (
        load_audio_subsample(
            file_path=(
                sample_information[
                    "mix_path"
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
            ),
            clip_duration=(
                CLIP_DURATION_SECONDS
            )
        )
    )

    patches = (
        process_audio_subsample(
            audio,
            sample_rate,
            data_representation=(
                data_representation
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

    patches = (
        patches
        .squeeze(0)
        .to(
            dtype=torch.float32
        )
    )

    label_vector = (
        sample_information[
            "label_vector"
        ]
    )

    if not isinstance(
        label_vector,
        torch.Tensor
    ):
        label_vector = torch.tensor(
            label_vector,
            dtype=torch.float32
        )

    label_vector = (
        label_vector
        .clone()
        .detach()
        .to(
            dtype=torch.float32
        )
    )

    if label_vector.ndim != 1:
        raise ValueError(
            "Expected one-dimensional "
            "multi-hot label vector."
        )

    if (
        label_vector.shape[0]
        != NUM_CLASSES
    ):
        raise ValueError(
            "Multi-hot label vector has "
            "incorrect size. "
            f"Expected {NUM_CLASSES}, "
            f"received "
            f"{label_vector.shape[0]}."
        )

    return (
        patches,
        label_vector
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def print_multilabel_sample(
    sample_information
):
    """
    Prints one indexed sample in a readable format.
    """

    print(
        f"Track: "
        f"{sample_information['track']}"
    )

    print(
        f"Time: "
        f"{sample_information['start_time']:.2f} "
        f"- "
        f"{sample_information['end_time']:.2f} "
        f"seconds"
    )

    print(
        f"Mix loudness: "
        f"{sample_information['mix_loudness_db']:.2f} "
        f"dBFS"
    )

    print(
        f"Labels: "
        f"{list(sample_information['labels'])}"
    )

    print(
        f"Label indices: "
        f"{list(sample_information['label_indices'])}"
    )

    print(
        "Multi-hot:"
    )

    print(
        sample_information[
            "label_vector"
        ]
    )

    print(
        "Active stems:"
    )

    for stem in (
        sample_information[
            "active_stems"
        ]
    ):

        print(
            f"    "
            f"{stem['stem_id']} "
            f"| "
            f"{stem['label']} "
            f"| "
            f"{stem['loudness_db']:.2f} dBFS"
        )


def print_multilabel_statistics(
    statistics
):
    """
    Prints the most useful dataset processing
    statistics.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-LABEL DATASET STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"Tracks found: "
        f"{statistics['tracks_found']}"
    )

    print(
        f"Valid tracks: "
        f"{statistics['valid_tracks']}"
    )

    print(
        f"Invalid tracks: "
        f"{statistics['invalid_tracks']}"
    )

    print(
        f"Total 5-second samples: "
        f"{statistics['total_sub_samples']}"
    )

    print(
        f"Valid labelled samples: "
        f"{statistics['valid_sub_samples']}"
    )

    print(
        f"Quiet mixture samples: "
        f"{statistics['quiet_mix_sub_samples']}"
    )

    print(
        f"Samples with no active labels: "
        f"{statistics['no_active_label_sub_samples']}"
    )

    print(
        "\nPer-label active sample counts:"
    )

    for label, count in (
        statistics[
            "sample_label_balance"
        ].items()
    ):

        print(
            f"    "
            f"{label}: "
            f"{count}"
        )

    combinations = sorted(
        statistics[
            "label_combination_balance"
        ].items(),
        key=lambda item: item[1],
        reverse=True
    )

    print(
        "\nMost common label combinations:"
    )

    for (
        label_combination,
        count
    ) in combinations[:20]:

        print(
            f"    "
            f"{list(label_combination)}: "
            f"{count}"
        )


# ---------------------------------------------------------------------
# Basic test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Change this to the required split before running.
    DATASET_PATH = Path(
        "C:/Uni/YearProject/datasets/"
        "slakh2100_redux_16k/train"
    )

    samples, statistics = (
        build_multilabel_sample_index(
            dataset_path=( 
                DATASET_PATH
            ),
            set_limit=True,
            maximum_tracks=2,
            show_progress=True
        )
    )

    print_multilabel_statistics(
        statistics
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FIRST MULTI-LABEL SAMPLES"
    )

    print(
        "=" * 70
    )

    for sample in samples[:5]:

        print_multilabel_sample(
            sample
        )

        print(
            "-" * 70
        )

    if len(samples) > 0:

        patches, labels = (
            load_multilabel_mix_sample(
                samples[0],
                data_representation=(
                    convert_to_mel_spectrogram
                )
            )
        )

        print(
            "\nLazy loaded patch shape:",
            patches.shape
        )

        print(
            "Lazy loaded label shape:",
            labels.shape
        )

        print(
            "Lazy loaded labels:",
            labels
        )