import torch
from data_procs_testing import compute_log_mel_spectrogram
from embeddings_testing import convert_to_patches
from transformer_architecture_testing import Transformer

def load_model():
    model = Transformer(
    patch_dim=1280,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=512,
    num_classes=4,
    mask_ratio=0.3
    )

    saved_model = torch.load(
        "audio_transformer_model.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        saved_model["model_parameters"]
    )

    class_names = saved_model["class_names"]

    return model, class_names

def process_audio_file(file_path):
    log_mel_spec = compute_log_mel_spectrogram(file_path)
    tensor = torch.from_numpy(log_mel_spec)
    patches = convert_to_patches(tensor)
    return patches

if __name__ == "__main__":
    example_file_path = "archive\overheads\Overhead Sample 3.wav"

    model, class_names = load_model()
    print("Model loaded successfully.")

    patches = process_audio_file(example_file_path)

    with torch.no_grad():
        predicted_index, probabilities = model.classify(patches)

    predicted_index = predicted_index.item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[0, predicted_index].item()

    print("Predicted class:", predicted_class)
    print(f"Confidence: {confidence * 100:.2f}%")