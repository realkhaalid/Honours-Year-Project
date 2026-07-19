import torch
from embeddings_testing import (
    convert_to_patches,
    add_positional_encoding
)
from encoder_testing import (
    split_heads,
    combine_heads,
    linear_projection,
    calc_attention,
    risidual_connection,
    layer_norm,
    forward_pass
)
from unsupervised_reconstruction_testing import (
    mask_patches,
    reconstruction_head,
    hidden_patch_reconstruction_loss
)
from supervised_fine_tuning_testing import (
    average_pooling,
    classification_head,
    predict_class,
    cross_entropy_loss
)
from generate_datasets_testing import (
    load_and_process_supervised_dataset,
    load_and_process_unsupervised_dataset
)

class Transformer:
    def __init__(
        self,
        patch_dim=1280,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_classes=4,
        mask_ratio=0.3
    ):
        # Initialize the Transformer model with the given parameters
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio

        # Initialize embedding weights and biases
        self.W_embedding = torch.randn(patch_dim, embedding_dim) * 0.01
        self.W_embedding.requires_grad_()
        self.b_embedding = torch.zeros(embedding_dim)
        self.b_embedding.requires_grad_()

        # query, key, value weights and biases for multi-head attention
        self.W_query = torch.randn(embedding_dim, embedding_dim) * 0.01
        self.W_query.requires_grad_()
        self.b_query = torch.zeros(embedding_dim)
        self.b_query.requires_grad_()

        self.W_key = torch.randn(embedding_dim, embedding_dim) * 0.01
        self.W_key.requires_grad_()
        self.b_key = torch.zeros(embedding_dim)
        self.b_key.requires_grad_()

        self.W_value = torch.randn(embedding_dim, embedding_dim) * 0.01
        self.W_value.requires_grad_()
        self.b_value = torch.zeros(embedding_dim)
        self.b_value.requires_grad_()

        self.W_output = torch.randn(embedding_dim, embedding_dim) * 0.01
        self.W_output.requires_grad_()
        self.b_output = torch.zeros(embedding_dim)
        self.b_output.requires_grad_()

        # Initialize layer normalization parameters
        self.gamma1 = torch.ones(embedding_dim)
        self.gamma1.requires_grad_()
        self.beta1 = torch.zeros(embedding_dim)
        self.beta1.requires_grad_()

        #Initialize feed-forward network parameters
        self.W_ff1 = torch.randn(embedding_dim, hidden_dim) * 0.01
        self.W_ff1.requires_grad_()
        self.b_ff1 = torch.zeros(hidden_dim)
        self.b_ff1.requires_grad_()

        self.W_ff2 = torch.randn(hidden_dim, embedding_dim) * 0.01
        self.W_ff2.requires_grad_()
        self.b_ff2 = torch.zeros(embedding_dim)
        self.b_ff2.requires_grad_()

        self.gamma2 = torch.ones(embedding_dim)
        self.gamma2.requires_grad_()
        self.beta2 = torch.zeros(embedding_dim)
        self.beta2.requires_grad_()

        # Initialize reconstruction parameters
        self.W_reconstruct = torch.randn(embedding_dim, patch_dim) * 0.01
        self.W_reconstruct.requires_grad_()
        self.b_reconstruct = torch.zeros(patch_dim)
        self.b_reconstruct.requires_grad_()

        # Initialize classification parameters
        self.W_classifier = torch.randn(embedding_dim, num_classes) * 0.01
        self.W_classifier.requires_grad_()
        self.b_classifier = torch.zeros(num_classes)
        self.b_classifier.requires_grad_()

    def parameters(self):
            return [
                self.W_embedding, self.b_embedding,

                self.W_query, self.b_query,
                self.W_key, self.b_key,
                self.W_value, self.b_value,

                self.W_output, self.b_output,

                self.gamma1, self.beta1,
                self.W_ff1, self.b_ff1,
                self.W_ff2, self.b_ff2,
                self.gamma2, self.beta2,

                self.W_reconstruct, self.b_reconstruct,

                self.W_classifier, self.b_classifier
            ]
    
    def state_dict(self):
        return {
        # Patch embedding
        "W_embedding": self.W_embedding,
        "b_embedding": self.b_embedding,

        # Multi-head attention
        "W_query": self.W_query,
        "b_query": self.b_query,
        "W_key": self.W_key,
        "b_key": self.b_key,
        "W_value": self.W_value,
        "b_value": self.b_value,
        "W_output": self.W_output,
        "b_output": self.b_output,

        # First layer normalisation
        "gamma1": self.gamma1,
        "beta1": self.beta1,

        # Feed-forward network
        "W_ff1": self.W_ff1,
        "b_ff1": self.b_ff1,
        "W_ff2": self.W_ff2,
        "b_ff2": self.b_ff2,

        # Second layer normalisation
        "gamma2": self.gamma2,
        "beta2": self.beta2,

        # Reconstruction head
        "W_reconstruct": self.W_reconstruct,
        "b_reconstruct": self.b_reconstruct,

        # Classification head
        "W_classifier": self.W_classifier,
        "b_classifier": self.b_classifier
        }
    
    def load_state_dict(self, saved_parameters):
        for parameter_name, saved_value in saved_parameters.items():
            current_parameter = getattr(
                self,
                parameter_name
            )
            with torch.no_grad():
                current_parameter.copy_(saved_value)
        
    def embed_patches(self, patches):
        embeddings = patches @ self.W_embedding + self.b_embedding
        embeddings_with_positional_encoding = add_positional_encoding(embeddings)
        return embeddings_with_positional_encoding
    
    def encoder_layer(self, input):
        query = linear_projection(input, self.W_query, self.b_query)
        key = linear_projection(input, self.W_key, self.b_key)
        value = linear_projection(input, self.W_value, self.b_value)
        
        query_heads = split_heads(query, self.num_heads)
        key_heads = split_heads(key, self.num_heads)
        value_heads = split_heads(value, self.num_heads)

        attention_output, attention_weights = calc_attention(
            query_heads,
            key_heads,
            value_heads,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads
        )

        combined_attention = combine_heads(attention_output)

        final_projection = linear_projection(
            combined_attention,
            self.W_output,
            self.b_output
        )

        rc_output = risidual_connection(input, final_projection)

        ln_output = layer_norm(
            rc_output,
            self.gamma1,
            self.beta1
        )

        ff_output = forward_pass(
            ln_output,
            self.W_ff1,
            self.b_ff1,
            self.W_ff2,
            self.b_ff2
        )

        ff_residual_output = risidual_connection(ln_output, ff_output)

        encoder_output = layer_norm(
            ff_residual_output,
            self.gamma2,
            self.beta2
        )

        return encoder_output, attention_weights

    def unsupervised_reconstruction(self, original_patches):
        masked_patches, mask = mask_patches(
            original_patches,
            mask_ratio=self.mask_ratio
        )
        
        embeddings_with_positional_encoding = self.embed_patches(masked_patches)
        encoder_output, attention_weights = self.encoder_layer(embeddings_with_positional_encoding)

        reconstructed_patches = reconstruction_head(
            encoder_output,
            self.W_reconstruct,
            self.b_reconstruct
        )

        loss = hidden_patch_reconstruction_loss(
            reconstructed_patches,
            original_patches,
            mask
        )

        return loss, reconstructed_patches, mask, attention_weights

        

    def supervised_fine_tuning(self, patches, labels):
        embeddings_with_positional_encoding = self.embed_patches(patches)
        encoder_output, attention_weights = self.encoder_layer(embeddings_with_positional_encoding)

        pooled_output = average_pooling(encoder_output)

        logits = classification_head(
            pooled_output,
            self.W_classifier,
            self.b_classifier
        )

        loss = cross_entropy_loss(logits, labels)

        predicted_class, probabilities = predict_class(logits)

        return loss, logits, predicted_class, probabilities, attention_weights
    
    def classify(self, patches):
        embeddings_with_positional_encoding = self.embed_patches(patches)
        encoder_output, attention_weights = self.encoder_layer(embeddings_with_positional_encoding)

        pooled_output = average_pooling(encoder_output)

        logits = classification_head(
            pooled_output,
            self.W_classifier,
            self.b_classifier
        )

        probabilities = torch.softmax(
        logits,
        dim=-1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=-1
        )

        return predicted_index, probabilities
    
if __name__ == "__main__":
    unsupervised_dataset_path = "archive"
    supervised_dataset_path = "archive"

    unsupervised_log_mel_specs = load_and_process_unsupervised_dataset(unsupervised_dataset_path)
    supervised_log_mel_specs, supervised_labels, encoding_map, label_map = load_and_process_supervised_dataset(supervised_dataset_path)

    print(f"Unsupervised dataset processed with {len(unsupervised_log_mel_specs)} log-mel spectrograms")
    print(f"Supervised dataset processed with {len(supervised_log_mel_specs)} log-mel spectrograms and {len(supervised_labels)} labels")

    unsupervised_sample_spectrogram = unsupervised_log_mel_specs[0]
    supervised_sample_spectrogram = supervised_log_mel_specs[0]
    sample_label = supervised_labels[0].unsqueeze(0)
    unsupervised_patches = convert_to_patches(unsupervised_sample_spectrogram)
    supervised_patches = convert_to_patches(supervised_sample_spectrogram)

    transformer = Transformer()
    
    unsupervised_loss, reconstructed_patches, mask, attention_weights = transformer.unsupervised_reconstruction(unsupervised_patches)
    supervised_loss, logits, predicted_class, probabilities, attention_weights = transformer.supervised_fine_tuning(supervised_patches, sample_label)

    print("Unsupervised reconstruction loss:", unsupervised_loss.item())
    print("Supervised fine-tuning loss:", supervised_loss.item())