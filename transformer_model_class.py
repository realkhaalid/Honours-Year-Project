import torch
from torch.utils.data import DataLoader
import math
from pathlib import Path
from embeddings_functions import add_positional_encoding
from data_processing_pipeline import (
    convert_to_stft_spectrogram,
    convert_to_mel_spectrogram,
    convert_to_cqt_spectrogram
)
from encoder_functions import (
    split_heads,
    combine_heads,
    linear_projection,
    calc_attention,
    residual_connection,
    layer_norm,
    forward_pass
)
from unsupervised_reconstruction_functions import (
    create_patch_mask,
    reconstruction_head,
    hidden_patch_reconstruction_loss
)
from supervised_fine_tuning_functions import (
    average_pooling,
    classification_head,
    predict_class,
    cross_entropy_loss
)
from construct_dataset_class import (
    create_datasets,
    SLAKH2100_REDUX_16K_TRAIN,
    SLAKH2100_REDUX_16K_VALIDATION,
    SLAKH2100_REDUX_16K_TEST
)

class Transformer:
    def __init__(
        self,
        sample_batch,
        embedding_dim=128,
        num_heads=4,
        hidden_dims=(512,),
        num_encoder_layers=2,
        num_classes=13,
        mask_ratio=0.3,
        activation="gelu"
    ):
        if sample_batch.ndim != 3:
            raise ValueError(
                "Expected sample batch shape "
                "[batch_size, number_of_patches, patch_dim], "
                f"but received {sample_batch.shape}."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_heads."
            )

        if num_encoder_layers < 1:
            raise ValueError(
                "num_encoder_layers must be at least 1."
            )

        if len(hidden_dims) < 1:
            raise ValueError(
                "hidden_dims must contain at least "
                "one hidden layer dimension."
            )

        self.patch_dim = sample_batch.shape[-1]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dims = tuple(hidden_dims)
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.activation = activation
        self.device = sample_batch.device
        self.dtype = sample_batch.dtype

        # Patch embedding
        self.W_embedding = self._initialize_weight(
            self.patch_dim,
            embedding_dim
        )

        self.b_embedding = self._initialize_bias(
            embedding_dim
        )

        # Learnable mask token
        self.mask_token = torch.zeros(
            embedding_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_encoder_layers):

            encoder_parameters = (
                self._initialize_encoder_layer()
            )

            self.encoder_layers.append(
                encoder_parameters
            )

        # Reconstruction head
        self.W_reconstruct = self._initialize_weight(
            embedding_dim,
            self.patch_dim
        )

        self.b_reconstruct = self._initialize_bias(
            self.patch_dim
        )

        # Classification head
        self.W_classifier = self._initialize_weight(
            embedding_dim,
            num_classes
        )

        self.b_classifier = self._initialize_bias(
            num_classes
        )

    def _initialize_weight(
        self,
        input_dim,
        output_dim
    ):
        """
        Initializes a trainable weight matrix.
        """

        scale = math.sqrt(
            2.0 / (
                input_dim
                + output_dim
            )
        )

        weight = (
            torch.randn(
                input_dim,
                output_dim,
                device=self.device,
                dtype=self.dtype
            )
            * scale
        )

        weight.requires_grad_()

        return weight

    def _initialize_bias(
        self,
        output_dim
    ):
        """
        Initializes a trainable bias vector.
        """

        bias = torch.zeros(
            output_dim,
            device=self.device,
            dtype=self.dtype
        )

        bias.requires_grad_()

        return bias

    def _initialize_feed_forward_parameters(
        self
    ):
        """
        Initializes a configurable feed-forward network.
        """

        dimensions = (
            [self.embedding_dim]
            + list(self.hidden_dims)
            + [self.embedding_dim]
        )

        weights = []
        biases = []

        for layer_index in range(
            len(dimensions) - 1
        ):

            weights.append(
                self._initialize_weight(
                    dimensions[layer_index],
                    dimensions[layer_index + 1]
                )
            )

            biases.append(
                self._initialize_bias(
                    dimensions[layer_index + 1]
                )
            )

        return weights, biases

    def _initialize_encoder_layer(
        self
    ):
        """
        Initializes the parameters for one Transformer encoder layer.
        """

        W_query = self._initialize_weight(
            self.embedding_dim,
            self.embedding_dim
        )

        b_query = self._initialize_bias(
            self.embedding_dim
        )

        W_key = self._initialize_weight(
            self.embedding_dim,
            self.embedding_dim
        )

        b_key = self._initialize_bias(
            self.embedding_dim
        )

        W_value = self._initialize_weight(
            self.embedding_dim,
            self.embedding_dim
        )

        b_value = self._initialize_bias(
            self.embedding_dim
        )

        W_output = self._initialize_weight(
            self.embedding_dim,
            self.embedding_dim
        )

        b_output = self._initialize_bias(
            self.embedding_dim
        )

        gamma1 = torch.ones(
            self.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        beta1 = torch.zeros(
            self.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        gamma2 = torch.ones(
            self.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        beta2 = torch.zeros(
            self.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        (
            ff_weights,
            ff_biases
        ) = (
            self._initialize_feed_forward_parameters()
        )

        return {
            "W_query": W_query,
            "b_query": b_query,

            "W_key": W_key,
            "b_key": b_key,

            "W_value": W_value,
            "b_value": b_value,

            "W_output": W_output,
            "b_output": b_output,

            "gamma1": gamma1,
            "beta1": beta1,

            "gamma2": gamma2,
            "beta2": beta2,

            "ff_weights": ff_weights,
            "ff_biases": ff_biases
        }

    def _encoder_parameters(
        self
    ):
        """
        Returns all trainable encoder parameters.
        """

        parameters = []

        for layer in self.encoder_layers:

            parameters.extend([
                layer["W_query"],
                layer["b_query"],

                layer["W_key"],
                layer["b_key"],

                layer["W_value"],
                layer["b_value"],

                layer["W_output"],
                layer["b_output"],

                layer["gamma1"],
                layer["beta1"],

                layer["gamma2"],
                layer["beta2"]
            ])

            parameters.extend(
                layer["ff_weights"]
            )

            parameters.extend(
                layer["ff_biases"]
            )

        return parameters

    def unsupervised_parameters(
        self
    ):
        """
        Returns parameters used during unsupervised pretraining.
        """

        return [
            self.W_embedding,
            self.b_embedding,
            self.mask_token,

            *self._encoder_parameters(),

            self.W_reconstruct,
            self.b_reconstruct
        ]

    def supervised_parameters(
        self
    ):
        """
        Returns parameters used during supervised fine-tuning.
        """

        return [
            self.W_embedding,
            self.b_embedding,

            *self._encoder_parameters(),

            self.W_classifier,
            self.b_classifier
        ]
    
    def state_dict(
        self
    ):
        """
        Returns all model parameters.
        """

        encoder_states = []

        for layer in self.encoder_layers:

            encoder_states.append({
                "W_query": layer["W_query"],
                "b_query": layer["b_query"],

                "W_key": layer["W_key"],
                "b_key": layer["b_key"],

                "W_value": layer["W_value"],
                "b_value": layer["b_value"],

                "W_output": layer["W_output"],
                "b_output": layer["b_output"],

                "gamma1": layer["gamma1"],
                "beta1": layer["beta1"],

                "gamma2": layer["gamma2"],
                "beta2": layer["beta2"],

                "ff_weights":
                    layer["ff_weights"],

                "ff_biases":
                    layer["ff_biases"]
            })

        return {
            "W_embedding":
                self.W_embedding,

            "b_embedding":
                self.b_embedding,

            "mask_token":
                self.mask_token,

            "encoder_layers":
                encoder_states,

            "W_reconstruct":
                self.W_reconstruct,

            "b_reconstruct":
                self.b_reconstruct,

            "W_classifier":
                self.W_classifier,

            "b_classifier":
                self.b_classifier
        }
    
    def load_state_dict(
        self,
        saved_parameters
    ):
        """
        Loads previously saved model parameters.
        """

        if (
            len(saved_parameters["encoder_layers"])
            != self.num_encoder_layers
        ):
            raise ValueError(
                "Saved model has a different number "
                "of encoder layers."
            )

        with torch.no_grad():

            self.W_embedding.copy_(
                saved_parameters[
                    "W_embedding"
                ]
            )

            self.b_embedding.copy_(
                saved_parameters[
                    "b_embedding"
                ]
            )

            self.mask_token.copy_(
                saved_parameters[
                    "mask_token"
                ]
            )

            self.W_reconstruct.copy_(
                saved_parameters[
                    "W_reconstruct"
                ]
            )

            self.b_reconstruct.copy_(
                saved_parameters[
                    "b_reconstruct"
                ]
            )

            self.W_classifier.copy_(
                saved_parameters[
                    "W_classifier"
                ]
            )

            self.b_classifier.copy_(
                saved_parameters[
                    "b_classifier"
                ]
            )

            for layer_index in range(
                self.num_encoder_layers
            ):

                current_layer = (
                    self.encoder_layers[
                        layer_index
                    ]
                )

                saved_layer = (
                    saved_parameters[
                        "encoder_layers"
                    ][layer_index]
                )

                parameter_names = [
                    "W_query",
                    "b_query",

                    "W_key",
                    "b_key",

                    "W_value",
                    "b_value",

                    "W_output",
                    "b_output",

                    "gamma1",
                    "beta1",

                    "gamma2",
                    "beta2"
                ]

                for parameter_name in (
                    parameter_names
                ):

                    current_layer[
                        parameter_name
                    ].copy_(
                        saved_layer[
                            parameter_name
                        ]
                    )

                for ff_index in range(
                    len(current_layer["ff_weights"])
                ):

                    current_layer[
                        "ff_weights"
                    ][ff_index].copy_(
                        saved_layer[
                            "ff_weights"
                        ][ff_index]
                    )

                    current_layer[
                        "ff_biases"
                    ][ff_index].copy_(
                        saved_layer[
                            "ff_biases"
                        ][ff_index]
                    )
        
    def embed_patches(
        self,
        patches,
        mask=None
    ):
        """
        Projects spectrogram patches into the
        embedding space and adds temporal position.
        """

        # Embed patches
        embeddings = (
            patches
            @ self.W_embedding
            + self.b_embedding
        )

        # Apply mask for unsupervised reconstruction
        if mask is not None:

            mask_expanded = (
                mask.unsqueeze(-1)
            )

            mask_token = (
                self.mask_token
                .view(
                    1,
                    1,
                    self.embedding_dim
                )
            )

            embeddings = torch.where(
                mask_expanded,
                mask_token,
                embeddings
            )

        # Add positional encoding
        embeddings = (
            add_positional_encoding(
                embeddings
            )
        )

        return embeddings
    
    def encoder_layer(
        self,
        input,
        parameters
    ):
        """
        Passes the input through one Transformer
        encoder using pre-layer normalization.
        """

        # Attention normalization
        normalized_input = layer_norm(
            input,
            parameters["gamma1"],
            parameters["beta1"]
        )

        # Query, key and value projections
        query = linear_projection(
            normalized_input,
            parameters["W_query"],
            parameters["b_query"]
        )

        key = linear_projection(
            normalized_input,
            parameters["W_key"],
            parameters["b_key"]
        )

        value = linear_projection(
            normalized_input,
            parameters["W_value"],
            parameters["b_value"]
        )

        # Multi-head attention
        query_heads = split_heads(
            query,
            self.num_heads
        )

        key_heads = split_heads(
            key,
            self.num_heads
        )

        value_heads = split_heads(
            value,
            self.num_heads
        )

        (
            attention_output,
            attention_weights
        ) = calc_attention(
            query_heads,
            key_heads,
            value_heads
        )

        combined_attention = combine_heads(
            attention_output
        )

        projected_attention = (
            linear_projection(
                combined_attention,
                parameters["W_output"],
                parameters["b_output"]
            )
        )

        # Attention residual connection
        attention_residual = (
            residual_connection(
                input,
                projected_attention
            )
        )

        # Feed-forward normalization
        normalized_attention = layer_norm(
            attention_residual,
            parameters["gamma2"],
            parameters["beta2"]
        )

        # Configurable feed-forward network
        ff_output = forward_pass(
            normalized_attention,
            parameters["ff_weights"],
            parameters["ff_biases"],
            activation=self.activation
        )

        # Feed-forward residual connection
        encoder_output = (
            residual_connection(
                attention_residual,
                ff_output
            )
        )

        return (
            encoder_output,
            attention_weights
        )

    def encoder_stack(
        self,
        input
    ):
        """
        Passes the spectrogram embeddings through
        every encoder layer sequentially.
        """

        encoder_output = input

        all_attention_weights = []

        for layer in self.encoder_layers:

            (
                encoder_output,
                attention_weights
            ) = self.encoder_layer(
                encoder_output,
                layer
            )

            all_attention_weights.append(
                attention_weights
            )

        return (
            encoder_output,
            all_attention_weights
        )

    def unsupervised_reconstruction(
        self,
        original_patches
    ):
        """
        Performs masked spectrogram patch
        reconstruction.
        """

        mask = create_patch_mask(
            original_patches,
            mask_ratio=self.mask_ratio
        )

        embeddings = self.embed_patches(
            original_patches,
            mask=mask
        )

        (
            encoder_output,
            attention_weights
        ) = self.encoder_stack(
            embeddings
        )

        reconstructed_patches = (
            reconstruction_head(
                encoder_output,
                self.W_reconstruct,
                self.b_reconstruct
            )
        )

        loss = (
            hidden_patch_reconstruction_loss(
                original_patches,
                reconstructed_patches,
                mask
            )
        )

        return (
            loss,
            reconstructed_patches,
            mask,
            attention_weights
        )

    def supervised_fine_tuning(
        self,
        patches,
        labels
    ):
        """
        Performs supervised instrument
        classification.
        """

        embeddings = self.embed_patches(
            patches
        )

        (
            encoder_output,
            attention_weights
        ) = self.encoder_stack(
            embeddings
        )

        pooled_output = average_pooling(
            encoder_output
        )

        logits = classification_head(
            pooled_output,
            self.W_classifier,
            self.b_classifier
        )

        loss = cross_entropy_loss(
            logits,
            labels
        )

        (
            predicted_class,
            probabilities
        ) = predict_class(
            logits
        )

        return (
            loss,
            logits,
            predicted_class,
            probabilities,
            attention_weights
        )
    
    def classify(
        self,
        patches
    ):
        """
        Performs instrument classification without
        calculating supervised loss.
        """

        embeddings = self.embed_patches(
            patches
        )

        (
            encoder_output,
            attention_weights
        ) = self.encoder_stack(
            embeddings
        )

        pooled_output = average_pooling(
            encoder_output
        )

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

        return (
            predicted_index,
            probabilities
        )
    
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
        training_path=SLAKH2100_REDUX_16K_TRAIN,
        validation_path=SLAKH2100_REDUX_16K_VALIDATION,
        test_path=SLAKH2100_REDUX_16K_TEST,
        data_representation=(
            convert_to_cqt_spectrogram
        ),
        set_limit=True,
        maximum_tracks=20
    )

    # Retrieve model label mappings
    (
        model_label_to_index,
        model_index_to_label
    ) = (
        supervised_training_dataset
        .return_label_mappings()
    )

    # Create DataLoaders
    supervised_training_loader = DataLoader(
        supervised_training_dataset,
        batch_size=16,
        shuffle=True
    )

    unsupervised_training_loader = DataLoader(
        unsupervised_training_dataset,
        batch_size=16,
        shuffle=True
    )

    # Get sample batch for model initialization
    (
        supervised_sample_batch,
        label_batch
    ) = next(
        iter(
            supervised_training_loader
        )
    )

    (
        unsupervised_sample_batch,
        _
    ) = next(
        iter(
            unsupervised_training_loader
        )
    )

    # Initialize Transformer
    transformer = Transformer(
        sample_batch=supervised_sample_batch,
        embedding_dim=128,
        num_heads=4,
        hidden_dims=(512,),
        num_encoder_layers=2,
        num_classes=len(
            model_label_to_index
        ),
        mask_ratio=0.3,
        activation="gelu"
    )

    # Count model parameters
    all_parameters = (
        transformer.unsupervised_parameters()
        + transformer.supervised_parameters()
    )

    unique_parameters = {
        id(parameter): parameter
        for parameter in all_parameters
    }

    total_parameters = sum(
        parameter.numel()
        for parameter
        in unique_parameters.values()
    )

    print(
        "\nTransformer Model"
    )

    print("=" * 60)

    print(
        f"Patch dimension: "
        f"{transformer.patch_dim}"
    )

    print(
        f"Embedding dimension: "
        f"{transformer.embedding_dim}"
    )

    print(
        f"Number of attention heads: "
        f"{transformer.num_heads}"
    )

    print(
        f"Number of encoder layers: "
        f"{transformer.num_encoder_layers}"
    )

    print(
        f"Feed-forward hidden dimensions: "
        f"{transformer.hidden_dims}"
    )

    print(
        f"Activation function: "
        f"{transformer.activation}"
    )

    print(
        f"Number of classes: "
        f"{transformer.num_classes}"
    )

    print(
        f"Total model parameters: "
        f"{total_parameters:,}"
    )

    # Check label mappings
    print(
        "\nLabel Mapping Checks"
    )

    print("=" * 60)

    print(
        "Model mapping matches validation mapping:",
        model_label_to_index
        == label_to_index_val
    )

    print(
        "Model mapping matches test mapping:",
        model_label_to_index
        == label_to_index_test
    )

    # Test unsupervised reconstruction
    (
        unsupervised_loss,
        reconstructed_patches,
        mask,
        unsupervised_attention
    ) = (
        transformer
        .unsupervised_reconstruction(
            unsupervised_sample_batch
        )
    )

    # Test supervised fine-tuning
    (
        supervised_loss,
        logits,
        predicted_class,
        probabilities,
        supervised_attention
    ) = (
        transformer
        .supervised_fine_tuning(
            supervised_sample_batch,
            label_batch
        )
    )

    # Print unsupervised test results
    print(
        "\nUnsupervised Reconstruction Test"
    )

    print("=" * 60)

    print(
        "Unsupervised sample batch shape:",
        unsupervised_sample_batch.shape
    )

    print(
        "Reconstructed patches shape:",
        reconstructed_patches.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Total masked patches in batch:",
        mask.sum().item()
    )

    print(
        "Masked patches per sample:",
        mask.sum(dim=1)
    )

    print(
        "Unsupervised reconstruction loss:",
        unsupervised_loss.item()
    )

    print(
        "Number of encoder attention outputs:",
        len(
            unsupervised_attention
        )
    )

    # Print supervised test results
    print(
        "\nSupervised Fine-Tuning Test"
    )

    print("=" * 60)

    print(
        "Supervised sample batch shape:",
        supervised_sample_batch.shape
    )

    print(
        "Label batch shape:",
        label_batch.shape
    )

    print(
        "Logits shape:",
        logits.shape
    )

    print(
        "Probabilities shape:",
        probabilities.shape
    )

    print(
        "Predicted classes shape:",
        predicted_class.shape
    )

    print(
        "Supervised batch loss:",
        supervised_loss.item()
    )

    # Inspect first sample in supervised batch
    first_label = (
        label_batch[0].item()
    )

    first_prediction = (
        predicted_class[0].item()
    )

    first_true_probability = (
        probabilities[
            0,
            first_label
        ].item()
    )

    print(
        "\nFirst Supervised Batch Sample"
    )

    print("-" * 60)

    print(
        "Sample shape:",
        supervised_sample_batch[0].shape
    )

    print(
        "True label index:",
        first_label
    )

    print(
        "True label name:",
        model_index_to_label[
            first_label
        ]
    )

    print(
        "Predicted class index:",
        first_prediction
    )

    print(
        "Predicted class name:",
        model_index_to_label[
            first_prediction
        ]
    )

    print(
        "True-class probability:",
        first_true_probability
    )

    print(
        "Logits:",
        logits[0]
    )

    print(
        "Probabilities:",
        probabilities[0]
    )

    print(
        "Number of encoder attention outputs:",
        len(
            supervised_attention
        )
    )