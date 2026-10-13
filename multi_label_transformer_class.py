import math

import torch

from embeddings_functions import add_positional_encoding

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
    classification_head
)


class MultiLabelTransformer:
    """
    Transformer model for multi-label musical instrument classification.

    Unsupervised pretraining is intentionally kept the same as the
    original Transformer and can use the original stem dataset.

    Supervised fine-tuning uses the new multi-label mix dataset.
    """

    def __init__(
        self,
        sample_batch,
        embedding_dim=128,
        num_heads=4,
        hidden_dims=(512,),
        num_encoder_layers=2,
        num_classes=13,
        mask_ratio=0.3,
        activation="gelu",
        classification_threshold=0.5
    ):
        if sample_batch.ndim != 3:
            raise ValueError(
                "Expected sample batch shape "
                "[batch_size, number_of_patches, patch_dim], "
                f"but received {sample_batch.shape}."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if num_encoder_layers < 1:
            raise ValueError(
                "num_encoder_layers must be at least 1."
            )

        if len(hidden_dims) < 1:
            raise ValueError(
                "hidden_dims must contain at least one hidden layer."
            )

        if num_classes < 1:
            raise ValueError(
                "num_classes must be at least 1."
            )

        if not 0.0 < classification_threshold < 1.0:
            raise ValueError(
                "classification_threshold must be between 0 and 1."
            )

        self.patch_dim = sample_batch.shape[-1]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dims = tuple(hidden_dims)
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.activation = activation
        self.classification_threshold = classification_threshold
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
            self.encoder_layers.append(
                self._initialize_encoder_layer()
            )

        # Reconstruction head
        self.W_reconstruct = self._initialize_weight(
            embedding_dim,
            self.patch_dim
        )

        self.b_reconstruct = self._initialize_bias(
            self.patch_dim
        )

        # Multi-label classification head
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
        Xavier-normal style initialization.
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
        Initializes a trainable zero bias vector.
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
        Initializes the feed-forward network parameters.
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

        return (
            weights,
            biases
        )


    def _initialize_encoder_layer(
        self
    ):
        """
        Initializes one pre-layer-normalized encoder layer.
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
        ) = self._initialize_feed_forward_parameters()

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
        Parameters updated during unsupervised reconstruction.

        This is the same parameter set as the original model.
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
        Parameters updated during supervised multi-label fine-tuning.
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

        The state layout is deliberately compatible with the original
        Transformer state dictionary.
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
                "ff_weights": layer["ff_weights"],
                "ff_biases": layer["ff_biases"]
            })

        return {
            "W_embedding": self.W_embedding,
            "b_embedding": self.b_embedding,
            "mask_token": self.mask_token,
            "encoder_layers": encoder_states,
            "W_reconstruct": self.W_reconstruct,
            "b_reconstruct": self.b_reconstruct,
            "W_classifier": self.W_classifier,
            "b_classifier": self.b_classifier
        }


    def _validate_saved_architecture(
        self,
        saved_parameters
    ):
        """
        Checks whether a saved state is compatible.
        """

        required_keys = {
            "W_embedding",
            "b_embedding",
            "mask_token",
            "encoder_layers",
            "W_reconstruct",
            "b_reconstruct",
            "W_classifier",
            "b_classifier"
        }

        missing_keys = (
            required_keys
            - set(saved_parameters.keys())
        )

        if missing_keys:
            raise KeyError(
                "Saved parameters are missing: "
                f"{sorted(missing_keys)}"
            )

        if (
            saved_parameters["W_embedding"].shape
            != self.W_embedding.shape
        ):
            raise ValueError(
                "Saved patch embedding shape does not match. "
                f"Saved: {saved_parameters['W_embedding'].shape}, "
                f"current: {self.W_embedding.shape}."
            )

        if (
            len(saved_parameters["encoder_layers"])
            != self.num_encoder_layers
        ):
            raise ValueError(
                "Saved model has a different number "
                "of encoder layers."
            )

        if (
            saved_parameters["W_reconstruct"].shape
            != self.W_reconstruct.shape
        ):
            raise ValueError(
                "Saved reconstruction head shape "
                "does not match the current model."
            )


    def _load_encoder_layers(
        self,
        saved_encoder_layers
    ):
        """
        Copies saved encoder parameters into this model.
        """

        for layer_index in range(
            self.num_encoder_layers
        ):
            current_layer = (
                self.encoder_layers[
                    layer_index
                ]
            )

            saved_layer = (
                saved_encoder_layers[
                    layer_index
                ]
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

            for parameter_name in parameter_names:
                current_layer[
                    parameter_name
                ].copy_(
                    saved_layer[
                        parameter_name
                    ]
                )

            if (
                len(current_layer["ff_weights"])
                != len(saved_layer["ff_weights"])
            ):
                raise ValueError(
                    "Saved feed-forward structure "
                    "does not match the current model."
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


    def load_state_dict(
        self,
        saved_parameters
    ):
        """
        Loads a complete multi-label model state.
        """

        self._validate_saved_architecture(
            saved_parameters
        )

        if (
            saved_parameters["W_classifier"].shape
            != self.W_classifier.shape
        ):
            raise ValueError(
                "Saved classification head shape "
                "does not match the current model."
            )

        with torch.no_grad():
            self.W_embedding.copy_(
                saved_parameters["W_embedding"]
            )

            self.b_embedding.copy_(
                saved_parameters["b_embedding"]
            )

            self.mask_token.copy_(
                saved_parameters["mask_token"]
            )

            self.W_reconstruct.copy_(
                saved_parameters["W_reconstruct"]
            )

            self.b_reconstruct.copy_(
                saved_parameters["b_reconstruct"]
            )

            self.W_classifier.copy_(
                saved_parameters["W_classifier"]
            )

            self.b_classifier.copy_(
                saved_parameters["b_classifier"]
            )

            self._load_encoder_layers(
                saved_parameters["encoder_layers"]
            )


    def load_pretrained_state_dict(
        self,
        saved_parameters,
        load_reconstruction_head=True
    ):
        """
        Loads only the pretrained representation.

        The multi-label classification head is deliberately not loaded,
        so it remains freshly initialized for multi-label fine-tuning.

        This can load a state produced by the original Transformer,
        provided the patch dimension and architecture match.
        """

        self._validate_saved_architecture(
            saved_parameters
        )

        with torch.no_grad():
            self.W_embedding.copy_(
                saved_parameters["W_embedding"]
            )

            self.b_embedding.copy_(
                saved_parameters["b_embedding"]
            )

            self.mask_token.copy_(
                saved_parameters["mask_token"]
            )

            self._load_encoder_layers(
                saved_parameters["encoder_layers"]
            )

            if load_reconstruction_head:
                self.W_reconstruct.copy_(
                    saved_parameters["W_reconstruct"]
                )

                self.b_reconstruct.copy_(
                    saved_parameters["b_reconstruct"]
                )


    def embed_patches(
        self,
        patches,
        mask=None
    ):
        """
        Projects patches into embedding space and adds position.
        """

        if patches.ndim != 3:
            raise ValueError(
                "Expected patches shape "
                "[batch_size, number_of_patches, patch_dim]."
            )

        if patches.shape[-1] != self.patch_dim:
            raise ValueError(
                "Patch dimension does not match model. "
                f"Expected {self.patch_dim}, "
                f"received {patches.shape[-1]}."
            )

        embeddings = (
            patches
            @ self.W_embedding
            + self.b_embedding
        )

        if mask is not None:
            if mask.shape != patches.shape[:2]:
                raise ValueError(
                    "Mask must have shape "
                    "[batch_size, number_of_patches]."
                )

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

        embeddings = add_positional_encoding(
            embeddings
        )

        return embeddings


    def encoder_layer(
        self,
        input,
        parameters
    ):
        """
        Passes input through one Transformer encoder layer.
        """

        normalized_input = layer_norm(
            input,
            parameters["gamma1"],
            parameters["beta1"]
        )

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

        projected_attention = linear_projection(
            combined_attention,
            parameters["W_output"],
            parameters["b_output"]
        )

        attention_residual = residual_connection(
            input,
            projected_attention
        )

        normalized_attention = layer_norm(
            attention_residual,
            parameters["gamma2"],
            parameters["beta2"]
        )

        ff_output = forward_pass(
            normalized_attention,
            parameters["ff_weights"],
            parameters["ff_biases"],
            activation=self.activation
        )

        encoder_output = residual_connection(
            attention_residual,
            ff_output
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
        Passes embeddings through every encoder layer.
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
        Performs masked spectrogram patch reconstruction.

        This is intentionally the same operation as the original
        single-label Transformer and should be fed the old stem dataset.
        """

        if original_patches.ndim != 3:
            raise ValueError(
                "Expected original_patches shape "
                "[batch_size, number_of_patches, patch_dim]."
            )

        if (
            original_patches.shape[-1]
            != self.patch_dim
        ):
            raise ValueError(
                "Unsupervised dataset patch dimension "
                "does not match model. "
                f"Expected {self.patch_dim}, "
                f"received {original_patches.shape[-1]}."
            )

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

        reconstructed_patches = reconstruction_head(
            encoder_output,
            self.W_reconstruct,
            self.b_reconstruct
        )

        loss = hidden_patch_reconstruction_loss(
            original_patches,
            reconstructed_patches,
            mask
        )

        return (
            loss,
            reconstructed_patches,
            mask,
            attention_weights
        )


    def binary_cross_entropy_with_logits(
        self,
        logits,
        labels
    ):
        """
        Numerically stable binary cross-entropy directly from logits.

        logits and labels:
            [batch_size, num_classes]
        """

        if logits.shape != labels.shape:
            raise ValueError(
                "Logits and labels must have the same shape. "
                f"Logits: {logits.shape}, "
                f"labels: {labels.shape}."
            )

        if logits.ndim != 2:
            raise ValueError(
                "Expected logits and labels shape "
                "[batch_size, num_classes]."
            )

        labels = labels.to(
            device=logits.device,
            dtype=logits.dtype
        )

        if not torch.all(
            (labels == 0.0)
            | (labels == 1.0)
        ):
            raise ValueError(
                "Multi-label targets must contain only 0 and 1."
            )

        element_loss = (
            torch.clamp(
                logits,
                min=0.0
            )
            - logits * labels
            + torch.log1p(
                torch.exp(
                    -torch.abs(
                        logits
                    )
                )
            )
        )

        return element_loss.mean()


    def predict_labels(
        self,
        logits,
        threshold=None
    ):
        """
        Applies sigmoid and thresholds each class independently.
        """

        if threshold is None:
            threshold = (
                self.classification_threshold
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = torch.sigmoid(
            logits
        )

        predicted_labels = (
            probabilities
            >= threshold
        ).to(
            dtype=torch.float32
        )

        return (
            predicted_labels,
            probabilities
        )


    def supervised_fine_tuning(
        self,
        patches,
        labels,
        threshold=None
    ):
        """
        Performs supervised multi-label classification.
        """

        if patches.ndim != 3:
            raise ValueError(
                "Expected patches shape "
                "[batch_size, number_of_patches, patch_dim]."
            )

        if labels.ndim != 2:
            raise ValueError(
                "Expected labels shape "
                "[batch_size, num_classes]."
            )

        if labels.shape[0] != patches.shape[0]:
            raise ValueError(
                "Patch and label batch sizes do not match."
            )

        if labels.shape[1] != self.num_classes:
            raise ValueError(
                "Target class dimension does not match model. "
                f"Expected {self.num_classes}, "
                f"received {labels.shape[1]}."
            )

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

        loss = self.binary_cross_entropy_with_logits(
            logits,
            labels
        )

        (
            predicted_labels,
            probabilities
        ) = self.predict_labels(
            logits,
            threshold=threshold
        )

        return (
            loss,
            logits,
            predicted_labels,
            probabilities,
            attention_weights
        )


    def classify(
        self,
        patches,
        threshold=None
    ):
        """
        Performs multi-label classification without loss.
        """

        embeddings = self.embed_patches(
            patches
        )

        (
            encoder_output,
            _
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

        (
            predicted_labels,
            probabilities
        ) = self.predict_labels(
            logits,
            threshold=threshold
        )

        return (
            predicted_labels,
            probabilities
        )


    def classify_with_logits(
        self,
        patches,
        threshold=None
    ):
        """
        Returns logits, binary predictions, probabilities and attention.
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

        (
            predicted_labels,
            probabilities
        ) = self.predict_labels(
            logits,
            threshold=threshold
        )

        return (
            logits,
            predicted_labels,
            probabilities,
            attention_weights
        )


if __name__ == "__main__":

    sample_batch = torch.randn(
        16,
        15,
        1280,
        dtype=torch.float32
    )

    label_batch = torch.randint(
        low=0,
        high=2,
        size=(16, 13)
    ).to(
        dtype=torch.float32
    )

    model = MultiLabelTransformer(
        sample_batch=sample_batch,
        embedding_dim=128,
        num_heads=4,
        hidden_dims=(256, 512, 256),
        num_encoder_layers=2,
        num_classes=13,
        mask_ratio=0.3,
        activation="relu",
        classification_threshold=0.5
    )

    (
        unsupervised_loss,
        reconstructed_patches,
        mask,
        unsupervised_attention
    ) = model.unsupervised_reconstruction(
        sample_batch
    )

    print("\nUnsupervised Reconstruction Test")
    print("=" * 70)
    print("Input shape:", sample_batch.shape)
    print(
        "Reconstructed shape:",
        reconstructed_patches.shape
    )
    print("Mask shape:", mask.shape)
    print(
        "Reconstruction loss:",
        unsupervised_loss.item()
    )
    print(
        "Encoder layers:",
        len(unsupervised_attention)
    )

    (
        supervised_loss,
        logits,
        predicted_labels,
        probabilities,
        supervised_attention
    ) = model.supervised_fine_tuning(
        sample_batch,
        label_batch
    )

    print("\nMulti-Label Supervised Test")
    print("=" * 70)
    print("Input shape:", sample_batch.shape)
    print("Target shape:", label_batch.shape)
    print("Target dtype:", label_batch.dtype)
    print("Logits shape:", logits.shape)
    print(
        "Probability shape:",
        probabilities.shape
    )
    print(
        "Predicted label shape:",
        predicted_labels.shape
    )
    print(
        "Supervised loss:",
        supervised_loss.item()
    )
    print("First target:", label_batch[0])
    print(
        "First probabilities:",
        probabilities[0]
    )
    print(
        "First predictions:",
        predicted_labels[0]
    )
    print(
        "Encoder layers:",
        len(supervised_attention)
    )