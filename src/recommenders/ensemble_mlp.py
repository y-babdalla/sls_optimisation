"""Main file for training and evaluating a NN ensemble on regression or classification tasks.

This file defines various components for training and evaluating a neural network
ensemble on regression or classification tasks. It includes functionality such
as:
- Custom dataset classes (masking for self-learning).
- Residual connections, attention, and embedding modules.
- An ensemble MLP and a PyTorch Lightning module for training and inference.
- Methods for performing classification, regression, and interpretability
  (via SHAP).

Run this file to see a demonstration of training on the 'moons' dataset, followed
by classification on the same data.
"""

import logging
import os
import pickle as pkl
import random

import numpy as np
import shap
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch import einsum
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Accuracy, MeanMetric, MeanSquaredError

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

CACHE_HIDDEN_KEY = "hidden_features"
CACHE_ATTENTION_KEY = "attention_weights"
CACHE_TARGET_KEY = "targets"
CACHE_PREDICTION_KEY = "predictions"


class SelfLearningMaskedRegressionDataset(Dataset):
    """Randomly mask a portion of non-zero features in the dataset."""

    def __init__(self, dataset: Dataset, dropout: float) -> None:
        """Initialise the dataset with a dropout rate for masking.

        Args:
            dataset: Original dataset, e.g. TensorDataset(x, y). Only x is used.
            dropout: Fraction of non-zero features to mask out.
        """
        super().__init__()
        self._dataset = dataset
        if not (0 <= dropout <= 1):
            msg = f"Invalid dropout rate: {dropout}. Must be between 0 and 1."
            raise ValueError(msg)
        self._dropout = dropout

    def __len__(self) -> int:
        # noinspection PyTypeChecker
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked input and original input as the target."""
        x, _ = self._dataset[index]
        x = x.clone()
        y = x.clone()

        non_zero_indices = x.nonzero(as_tuple=True)
        # Create a boolean mask for dropping out some fraction of non-zero features
        mask_fraction = ~(torch.rand(len(non_zero_indices[0])) < self._dropout)
        mask_fraction = mask_fraction.float()

        # Apply the mask
        for idx, mask_val in zip(non_zero_indices[0], mask_fraction, strict=False):
            x[idx] = x[idx] * mask_val

        return x, y


class Add(nn.Module):
    """Residual addition layer."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialise the layer.

        Args:
            inplace: If True, perform in-place addition for improved memory efficiency.
        """
        super().__init__()
        self._inplace = inplace

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors, possibly in place."""
        return x.add_(y) if self._inplace else x.add(y)


class Reduction(nn.Module):
    """Reduction layer supporting mean, max, or sum."""

    def __init__(self, dim: int, reduction: str) -> None:
        """Initialise the reduction layer.

        Args:
            dim: Dimension over which to reduce.
            reduction: Type of reduction ('mean', 'max', or 'sum').
        """
        super().__init__()
        if reduction not in {"mean", "max", "sum"}:
            msg = f"Invalid reduction type: {reduction}. Must be 'mean', 'max', or 'sum'."
            raise ValueError(msg)

        self._dim = dim
        self._reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the chosen reduction over the given dimension."""
        if self._reduction == "mean":
            return x.mean(dim=self._dim)
        if self._reduction == "max":
            return x.max(dim=self._dim)[0]
        if self._reduction == "sum":
            return x.sum(dim=self._dim)
        raise NotImplementedError(f"Reduction not implemented: {self._reduction}")

    def extra_repr(self) -> str:
        """Return a string representation of the layer."""
        return f"dim={self._dim}, reduction={self._reduction}"


class Attention(nn.Module):
    """Multi-head self-attention. Optionally reduces over num_nodes."""

    def __init__(
        self, input_dim: int, inner_dim: int, output_dim: int, num_heads: int, dropout: float
    ) -> None:
        """Initialise the attention layer.

        Args:
            input_dim: Dimensionality of the input features.
            inner_dim: Dimensionality per attention head.
            output_dim: Dimensionality of the final output representation.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability within the attention computations.
        """
        super().__init__()
        head_dim = inner_dim
        combined_inner = inner_dim * num_heads
        self._num_heads = num_heads
        self._scale = head_dim**-0.5
        self._reduction = Reduction(dim=1, reduction="sum")

        self._qkv = nn.Linear(input_dim, combined_inner * 3, bias=False)
        self._dropout = nn.Dropout(dropout)
        self._weights = nn.Parameter(torch.randn(num_heads, num_heads))
        self._norm = nn.Sequential(
            Rearrange("b h i j -> b i j h"),
            nn.LayerNorm(num_heads),
            Rearrange("b i j h -> b h i j"),
        )
        self._output = nn.Sequential(nn.Linear(combined_inner, output_dim), nn.Dropout(dropout))
        self._softmax = nn.Softmax(dim=-1)
        self._add_norm = nn.LayerNorm(output_dim)
        self._add = Add()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention and apply residual connection."""
        x_ = x
        if len(x_.shape) == 2:
            # Expand to (batch, num_nodes=..., features)
            x_ = x_.unsqueeze(-1)

        # Compute query, key, value
        qkv = self._qkv(x_).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self._num_heads) for t in qkv)

        # Scaled dot-product attention
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self._scale
        attention = self._softmax(dots)
        attention = self._dropout(attention)
        attention = einsum("b h i j, h g -> b g i j", attention, self._weights)
        attention = self._norm(attention)

        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self._output(out)

        # If original input was 2D, apply sum-reduction
        if len(x.shape) == 2:
            out = self._reduction(out)

        out = self._add_norm(out)
        out = self._add(x, out)

        # If original input was 3D, sum over the node dimension for final output
        if len(x.shape) == 3:
            out = out.sum(dim=1)

        return out, attention


class ResidualLinear(nn.Module):
    """Residual block with a linear transformation, dropout, ReLU, normalisation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the residual block.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            bias: Whether to include a bias term.
            norm: Normalisation type: 'batch', 'layer', or None.
            dropout: Dropout rate.
        """
        super().__init__()
        if norm not in {None, "batch", "layer"}:
            msg = f"Invalid normalisation: {norm}. Must be 'batch', 'layer', or None."
            raise ValueError(msg)

        self._linear = nn.Linear(in_features, out_features, bias=bias)
        self._relu = nn.ReLU()
        self._add = Add()
        if norm == "batch":
            self._norm = nn.BatchNorm1d(out_features)
        elif norm == "layer":
            self._norm = nn.LayerNorm(out_features)
        else:
            self._norm = nn.Identity()
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform linear transform -> dropout -> ReLU -> norm -> residual add."""
        y = self._linear(x)
        y = self._dropout(y)
        y = self._relu(y)
        y = self._norm(y)
        return self._add(x, y)


class MaterialEmbedding(nn.Module):
    """Learnable embeddings for materials plus a linear transform for proportions."""

    def __init__(self, num_materials: int, embedding_dim: int, reduce: bool = True) -> None:
        """Initialise the material embedding layer.

        Args:
            num_materials: Total number of distinct materials.
            embedding_dim: Embedding dimension for each material plus proportion linear transform.
            reduce: If True, sum over the materials dimension, producing (batch, embedding_dim).
        """
        super().__init__()
        self._embedding = nn.Embedding(num_materials, embedding_dim)
        self._proportions = nn.Linear(1, embedding_dim)
        self._reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the material embedding layer.

        Args:
            x: (batch_size, num_materials, 2) - last dim has [material_index, proportion].
        """
        mat_indices = x[..., 0].long()
        proportions = x[..., 1].unsqueeze(-1)
        mat_embeddings = self._embedding(mat_indices)
        prop_embeddings = self._proportions(proportions)
        out = mat_embeddings + prop_embeddings
        if self._reduce:
            out = out.sum(dim=1)
        return out


class ResidualMLP(nn.Module):
    """Residual multi-layer perceptron with optional embedding/attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int],
        bias: bool = True,
        dropout: float = 0.0,
        norm: str | None = None,
        embedding: bool = False,
        embedding_num_items: int = 100,
        attention: bool = True,
        attention_inner_dim: int = 64,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
    ) -> None:
        """Initialise the MLP.

        Args:
            in_features: Input dimension (or embedding dim if `embedding` is True).
            out_features: Final output dimension.
            hidden_features: List of dimensions for hidden layers.
            bias: Whether to include bias in linear layers.
            dropout: Dropout rate.
            norm: Normalisation type: 'batch', 'layer', or None.
            embedding: Whether input is (batch, num_materials, 2) for material embeddings.
            embedding_num_items: Number of items (material IDs) if embedding is True.
            attention: Whether to use a self-attention layer after embedding.
            attention_inner_dim: Dimensionality per attention head.
            attention_num_heads: Number of attention heads.
            attention_dropout: Dropout rate in the attention layer.
        """
        super().__init__()
        if norm not in {None, "batch", "layer"}:
            msg = f"Invalid normalization: {norm}. Must be 'batch', 'layer' or None."
            raise ValueError(msg)

        self._cache_hidden = None
        self._cache_attention = None
        self._cache_prediction = None

        # Set up embedding or pass-through
        if embedding:
            self._embedding = MaterialEmbedding(
                embedding_num_items, in_features, reduce=not attention
            )
            if attention:
                self._attention = Attention(
                    input_dim=in_features,
                    inner_dim=attention_inner_dim,
                    output_dim=in_features,
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                )
            else:
                self._attention = None
        else:
            self._embedding = nn.Identity()
            if attention:
                # In this scenario, we assume the input is (batch, 1)
                self._attention = Attention(
                    input_dim=1,
                    inner_dim=attention_inner_dim,
                    output_dim=in_features,
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                )
            else:
                self._attention = None

        # First linear layer, activation, and norm
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, hidden_features[0], bias=bias))
        layers.append(nn.ReLU())
        if norm == "batch":
            layers.append(nn.BatchNorm1d(hidden_features[0]))
        elif norm == "layer":
            layers.append(nn.LayerNorm(hidden_features[0]))

        # Additional hidden layers as residual blocks
        for idx in range(1, len(hidden_features)):
            layers.append(
                ResidualLinear(
                    hidden_features[idx - 1],
                    hidden_features[idx],
                    bias=bias,
                    norm=norm,
                    dropout=dropout,
                )
            )

        # Final output layer (no activation here)
        layers.append(nn.Linear(hidden_features[-1], out_features, bias=bias))
        self._layers = layers

    def reset_cache(self) -> None:
        """Clear cached hidden states, attention, and predictions."""
        self._cache_hidden = None
        self._cache_attention = None
        self._cache_prediction = None

    def forward(self, x: torch.Tensor, cache: bool = False) -> torch.Tensor:
        """Compute forward pass, optionally caching last hidden, attention, and output."""
        # Embedding step
        x = self._embedding(x)
        if self._attention is not None:
            x, attention = self._attention(x)
            if cache:
                if self._cache_attention is None:
                    self._cache_attention = attention.cpu().detach()
                else:
                    self._cache_attention = torch.cat(
                        [self._cache_attention, attention.cpu().detach()], dim=0
                    )

        # Pass through MLP layers
        for idx, layer in enumerate(self._layers):
            x = layer(x)
            # Cache the last hidden layer
            if cache and idx == len(self._layers) - 2:
                cached_x = x.cpu().detach()
                if self._cache_hidden is None:
                    self._cache_hidden = cached_x
                else:
                    self._cache_hidden = torch.cat([self._cache_hidden, cached_x], dim=0)

        if cache:
            cached_pred = x.cpu().detach()
            if self._cache_prediction is None:
                self._cache_prediction = cached_pred
            else:
                self._cache_prediction = torch.cat([self._cache_prediction, cached_pred], dim=0)
        return x

    def replace_output_layer(self, out_features: int) -> None:
        """Swap out the final output layer with a new linear layer of dimension out_features."""
        self._layers[-1] = nn.Linear(self._layers[-2].out_features, out_features)


class EnsembleMLP(nn.Module):
    """Ensemble of MLP models with a single trainable 'active' member at a time."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int],
        bias: bool = True,
        norm: str | None = None,
        dropout: float = 0.0,
        embedding: bool = False,
        embedding_num_items: int = 100,
        attention: bool = True,
        attention_inner_dim: int = 64,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
        num_members: int = 5,
        task: str = "regression",
    ) -> None:
        """Initialise the ensemble model.

        Args:
            in_features: Input dimensionality for each model.
            out_features: Output dimensionality for each model.
            hidden_features: List of hidden layer sizes.
            bias: If True, linear layers have a bias.
            norm: Normalisation type for layers: 'batch', 'layer', or None.
            dropout: Dropout probability.
            embedding: Whether to use a material-embedding approach.
            embedding_num_items: Number of material indices for embedding.
            attention: Whether to apply attention to the embedded features.
            attention_inner_dim: Dimensionality per attention head.
            attention_num_heads: Number of attention heads.
            attention_dropout: Dropout rate for attention.
            num_members: How many MLP models to keep in the ensemble.
            task: 'regression' or 'classification'.
        """
        super().__init__()
        if num_members < 1:
            raise ValueError("The number of ensemble members must be >= 1.")
        self._models = nn.ModuleList(
            [
                ResidualMLP(
                    in_features,
                    out_features,
                    hidden_features,
                    bias,
                    dropout,
                    norm,
                    embedding,
                    embedding_num_items,
                    attention,
                    attention_inner_dim,
                    attention_num_heads,
                    attention_dropout,
                )
                for _ in range(num_members)
            ]
        )
        self._num_members = num_members
        self._currently_trained_member = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._attention = attention
        self.set_task(task)

    def set_task(self, task: str) -> None:
        """Assign task type: 'regression' or 'classification'."""
        if task not in {"regression", "classification"}:
            raise ValueError(f"Unknown task {task}. Must be 'regression' or 'classification'.")
        self._task = task

    def reset_cache(self) -> None:
        """Clear cached hidden states/attention/predictions for all models."""
        for model in self._models:
            model.reset_cache()

    def reset_model(self) -> None:
        """Reset index tracking which member is currently trained."""
        self._currently_trained_member.data = torch.zeros_like(self._currently_trained_member.data)

    def forward(
        self, x: torch.Tensor, cache: bool = False, model_index: int | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass logic.

        - If training, only the active model is used.
        - If evaluating, either a single model (model_index) or average of all trained models is
            used.
        For regression: returns mean, std if multiple models are used.
        For classification: returns sigmoid of average logits.
        """
        if self.training:
            # Single model for training
            idx = self._currently_trained_member.item()
            if self._task == "classification":
                return torch.sigmoid(self._models[idx](x, cache))
            return self._models[idx](x, cache)

        # Inference mode
        if model_index is not None:
            if self._task == "classification":
                return torch.sigmoid(self._models[model_index](x, cache))
            return self._models[model_index](x, cache)

        # Use all models up to the active index
        up_to = self._currently_trained_member.item() + 1
        outputs = [self._models[i](x, cache) for i in range(up_to)]

        if self._task == "classification":
            sigmoids = [torch.sigmoid(o) for o in outputs]
            stacked = torch.stack(sigmoids, dim=0)
            return torch.mean(stacked, dim=0)
        stacked = torch.stack(outputs, dim=0)
        mean = torch.mean(stacked, dim=0)
        std_dev = torch.std(stacked, dim=0)
        return mean, std_dev

    def increment_current_member(self) -> None:
        """Move to the next ensemble member for training."""
        self._currently_trained_member.data.add_(1)

    def parameters(self, recurse: bool = True) -> torch.Tensor:
        """Override nn.Module.parameters() to return parameters of the active model."""
        idx = self._currently_trained_member.item()
        return self._models[idx].parameters(recurse=recurse)

    def replace_output_layer(self, out_features: int) -> None:
        """Replace the final layer in all ensemble members with a new linear layer."""
        for model in self._models:
            model.replace_output_layer(out_features)

    def get_cache(self) -> dict[str, torch.Tensor | None]:
        """Retrieve stacked caches for hidden features, attention, predictions."""
        hidden_list = []
        att_list = []
        pred_list = []
        for model in self._models:
            hidden_list.append(model._cache_hidden)
            att_list.append(model._cache_attention)
            pred_list.append(model._cache_prediction)
        hidden = (
            torch.stack(hidden_list, dim=0) if all(h is not None for h in hidden_list) else None
        )
        attention = torch.stack(att_list, dim=0) if all(a is not None for a in att_list) else None
        prediction = (
            torch.stack(pred_list, dim=0) if all(p is not None for p in pred_list) else None
        )
        return {
            CACHE_HIDDEN_KEY: hidden,
            CACHE_ATTENTION_KEY: attention,
            CACHE_PREDICTION_KEY: prediction,
        }


class EnsembleLearner(LightningModule):
    """PyTorch Lightning wrapper for training an EnsembleMLP with BCELoss or MSELoss."""

    def __init__(
        self,
        model: EnsembleMLP,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        task: str = "classification",
    ) -> None:
        """Initialise the ensemble learner.

        Args:
            model: The ensemble model.
            learning_rate: Optimiser learning rate.
            weight_decay: Weight decay for the optimiser.
            epochs: Number of training epochs.
            task: 'classification' or 'regression'.
        """
        super().__init__()
        self._model = model
        self._task = task
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        if task == "classification":
            self._loss = nn.BCELoss()
        else:
            self._loss = nn.MSELoss()

        self.save_hyperparameters(ignore=["model", "task"])
        self._create_metrics()

    def _create_metrics(self) -> None:
        """Instantiate metrics for loss and accuracy or MSE, for training/validation/testing."""
        self._loss_metrics = nn.ModuleDict(
            {"val": MeanMetric(), "test": MeanMetric(), "tra": MeanMetric()}
        )
        if self._task == "classification":
            self._accuracy_metrics = nn.ModuleDict(
                {
                    "val": Accuracy(task="binary", num_classes=2),
                    "test": Accuracy(task="binary", num_classes=2),
                    "tra": Accuracy(task="binary", num_classes=2),
                }
            )
        else:
            self._accuracy_metrics = nn.ModuleDict(
                {"val": MeanSquaredError(), "test": MeanSquaredError(), "tra": MeanSquaredError()}
            )

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.CosineAnnealingLR]]:
        """Configure optimiser and scheduler."""
        optimiser = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.hparams.epochs, eta_min=0.0
        )
        return [optimiser], [scheduler]

    def on_fit_start(self) -> None:
        """Move metrics to the device before training."""
        super().on_fit_start()
        self._loss_metrics.to(self.device)
        self._accuracy_metrics.to(self.device)

    def on_test_start(self) -> None:
        """Move metrics to the device before testing."""
        super().on_test_start()
        self._loss_metrics.to(self.device)
        self._accuracy_metrics.to(self.device)

    def increment_current_member(self) -> None:
        """Instruct the ensemble to move to the next member."""
        self._model.increment_current_member()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], phase: str) -> torch.Tensor:
        """Compute forward pass, loss, and update metrics for a given phase."""
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)
        self.log(f"{phase}/loss", loss, prog_bar=True)
        if self._task == "classification":
            pred_binary = torch.round(y_hat).long()
            self._accuracy_metrics[phase](pred_binary.squeeze(), y.squeeze())
        else:
            self._accuracy_metrics[phase](y_hat.squeeze(), y.squeeze())
        self._loss_metrics[phase](loss)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for each batch."""
        assert isinstance(batch_idx, int)
        return self._step(batch, "tra")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for each batch."""
        assert isinstance(batch_idx, int)

        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Testing step for each batch."""
        assert isinstance(batch_idx, int)

        return self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Log training metrics."""
        super().on_train_epoch_end()
        self.log("train/loss", self._loss_metrics["tra"].compute(), logger=True)
        self.log("train/acc", self._accuracy_metrics["tra"].compute(), logger=True)
        self._loss_metrics["tra"].reset()
        self._accuracy_metrics["tra"].reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        super().on_validation_epoch_end()
        self.log("val/loss", self._loss_metrics["val"].compute(), logger=True)
        self.log("val/acc", self._accuracy_metrics["val"].compute(), logger=True)
        self._loss_metrics["val"].reset()
        self._accuracy_metrics["val"].reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        super().on_test_epoch_end()
        self.log("test/loss", self._loss_metrics["test"].compute(), logger=True)
        self.log("test/acc", self._accuracy_metrics["test"].compute(), logger=True)
        self._loss_metrics["test"].reset()
        self._accuracy_metrics["test"].reset()


class Ensemble:
    """Scikit-learn-like wrapper for training, predicting, and interpreting an EnsembleMLP."""

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        hidden_features: list[int] = [32, 32],
        bias: bool = True,
        norm: str | None = None,
        dropout: float = 0.1,
        embedding: bool = True,
        embedding_num_items: int = 100,
        attention: bool = True,
        attention_inner_dim: int = 64,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
        num_members: int = 5,
        model: EnsembleMLP | None = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 32,
        seed: int = 42,
        gpu: int | None = None,
        logger: bool = False,
        checkpoint_dir: str = "checkpoints",
        regression_dropout: float = 0.3,
        task: str = "classification",
    ) -> None:
        """Initialise the ensemble model.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality.
            hidden_features: Sizes of the hidden layers.
            bias: Whether linear layers have a bias.
            norm: Normalisation type for hidden layers.
            dropout: Dropout rate for the MLP.
            embedding: Whether to use an embedding approach for the inputs.
            embedding_num_items: Number of embedding items if embedding is True.
            attention: Whether to enable a self-attention module.
            attention_inner_dim: Dimensionality per attention head.
            attention_num_heads: Number of attention heads.
            attention_dropout: Attention dropout rate.
            num_members: Number of MLP submodels in the ensemble.
            model: Optionally provide a pre-initialised EnsembleMLP.
            learning_rate: Optimiser learning rate.
            weight_decay: Weight decay for AdamW.
            epochs: Training epochs per ensemble member.
            batch_size: Training batch size.
            seed: Random seed.
            gpu: GPU index or None for CPU.
            logger: Enable or disable PyTorch Lightning logger.
            checkpoint_dir: Path for saving checkpoints (unused here).
            regression_dropout: Dropout rate if used in a self-learning masked regression scenario.
            task: 'regression' or 'classification'.
        """
        self._in_features = in_features
        self._out_features = out_features
        self._hidden_features = hidden_features
        self._bias = bias
        self._norm = norm
        self._dropout = dropout
        self._embedding = embedding
        self._embedding_num_items = embedding_num_items
        self._attention = attention
        self._attention_inner_dim = attention_inner_dim
        self._attention_num_heads = attention_num_heads
        self._attention_dropout = attention_dropout
        self._num_members = num_members
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._batch_size = batch_size
        self._seed = seed
        self._gpu = gpu
        self._regression_dropout = regression_dropout
        self._task = task
        self._logger = logger

        seed_everything(seed, workers=True)

        if model is None:
            self._model = EnsembleMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                bias=bias,
                norm=norm,
                dropout=dropout,
                embedding=embedding,
                embedding_num_items=embedding_num_items,
                attention=attention,
                attention_inner_dim=attention_inner_dim,
                attention_num_heads=attention_num_heads,
                attention_dropout=attention_dropout,
                num_members=num_members,
                task=task,
            )
        else:
            self._model = model

        self._learner = EnsembleLearner(
            self._model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            task=task,
        )

    def _prepare_data_loader(
        self, x: np.ndarray, y: np.ndarray, shuffle: bool = False
    ) -> DataLoader:
        """Convert x, y to tensors and return a DataLoader."""
        x_t = self._convert_to_tensor(x, torch.float)
        y_t = self._convert_to_tensor(y, torch.float)
        dataset = TensorDataset(x_t, y_t)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, num_workers=0)

    def _convert_to_tensor(
        self, x: np.ndarray | torch.Tensor, dtype: torch.dtype, device: torch.device | None = None
    ) -> torch.Tensor:
        """Convert a NumPy array or Tensor to a torch.Tensor on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(dtype)
        if device is not None:
            x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return x

    def _train_ensemble_member(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> None:
        """Train the active ensemble member until early stopping or epochs finish."""
        early_stop_callback = EarlyStopping(
            monitor="tra/loss", patience=10, verbose=True, mode="min"
        )
        trainer = Trainer(
            max_epochs=self._epochs,
            callbacks=[early_stop_callback],
            devices=[self._gpu] if self._gpu is not None else 1,
            logger=self._logger,
            accelerator="gpu" if self._gpu is not None else "cpu",
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        trainer.fit(self._learner, train_loader, val_loader)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train each member of the ensemble sequentially."""
        train_loader = self._prepare_data_loader(x_train, y_train, shuffle=True)
        val_loader = None
        if x_val is not None and y_val is not None:
            val_loader = self._prepare_data_loader(x_val, y_val, shuffle=False)

        for i in range(self._num_members):
            self._train_ensemble_member(train_loader, val_loader)
            if i < self._num_members - 1:
                self._learner.increment_current_member()

    def predict_proba(
        self,
        x: np.ndarray | torch.Tensor,
        cache: bool = False,
        return_tensor: bool = False,
        model_index: int | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Predict probabilities for classification or raw predictions for regression.

        If classification, the ensemble average (sigmoid) is returned if multiple models.
        """
        self._model.eval()
        x_t = self._convert_to_tensor(x, dtype=torch.float, device=self._learner.device)
        out = self._model(x_t, cache=cache, model_index=model_index)
        # For classification, out is shape (batch,)
        # For regression with multiple models, out is (mean, std).
        if isinstance(out, tuple):  # noqa: SIM108
            # Means, std dev for regression
            # Only returning means as "proba"-like structure
            out = out[0].flatten()
        else:
            out = out.flatten()

        if return_tensor:
            return out
        return out.cpu().detach().numpy()

    def predict(
        self, x: np.ndarray, cache: bool = False, model_index: int | None = None
    ) -> np.ndarray:
        """Predict class labels for classification or (mean, std) for regression."""
        if self._task == "classification":
            probabilities = self.predict_proba(x, cache=cache, model_index=model_index)
            preds = np.where(probabilities >= 0.5, 1, 0)
            return preds
        self._model.eval()
        x_t = self._convert_to_tensor(x, dtype=torch.float, device=self._learner.device)
        with torch.no_grad():
            out = self._model(x_t, cache=cache, model_index=model_index)
        mean, std_dev = out
        return mean.cpu().numpy(), std_dev.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state_dict to disk."""
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state_dict from disk."""
        self._model.load_state_dict(torch.load(path))

    def get_model(self) -> EnsembleMLP:
        """Return the raw ensemble model."""
        return self._model

    def get_cache(self) -> dict[str, torch.Tensor | None]:
        """Obtain any cached intermediate values from the ensemble."""
        return self._model.get_cache()

    def reset_cache(self) -> None:
        """Clear any cached intermediate values in the ensemble."""
        self._model.reset_cache()

    def interpret(
        self, x_train: np.ndarray, x_test: np.ndarray, sample_size: int = 50
    ) -> dict[str, np.ndarray | dict[int, np.ndarray]]:
        """Calculate SHAP values using a random background from x_train."""
        x_train_t = self._convert_to_tensor(x_train, torch.float32, self._learner.device)
        x_test_t = self._convert_to_tensor(x_test, torch.float32, self._learner.device)
        num_samples = min(sample_size, x_train_t.shape[0])
        background_indices = random.sample(range(x_train_t.shape[0]), num_samples)
        background = x_train_t[background_indices]

        # Calculate SHAP for each individual model
        shap_values_individual = {}
        for i, model_part in enumerate(self._model._models):
            explainer = shap.DeepExplainer(model_part, background)
            shap_values = explainer.shap_values(x_test_t)
            shap_values_individual[i] = shap_values

        # Calculate SHAP for the entire ensemble
        explainer_ensemble = shap.DeepExplainer(self._model, background)
        shap_values_ensemble = explainer_ensemble.shap_values(x_test_t)

        return {"mlp": shap_values_individual, "ensemble": shap_values_ensemble}


def quantize(
    x: np.ndarray, bins: int, min_value: float = 0.0, max_value: float = 1.0
) -> np.ndarray:
    """Map each row (feature1, feature2) to a bin index, then store it alongside proportion=1."""
    y = np.zeros((x.shape[0],))
    x_feature_1 = np.digitize(x[:, 0], np.linspace(min_value, max_value, bins)) - 1
    x_feature_2 = np.digitize(x[:, 1], np.linspace(min_value, max_value, bins)) - 1
    y = x_feature_1 + x_feature_2 * bins
    y = np.concatenate([y[:, None], np.ones((y.shape[0], 1))], axis=1)[:, np.newaxis, :]
    return y.astype(int)


def make_random_data(
    num_samples: int, num_features: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random input data and integer labels."""
    x_data = np.random.randn(num_samples, num_features)
    y_data = np.random.randint(0, num_classes, size=(num_samples,))
    return x_data, y_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    EMBEDDING = False
    ATTENTION = False
    ATTENTION_INNER_DIM = 64
    ATTENTION_NUM_HEADS = 4
    ATTENTION_DROPOUT = 0.1
    BINS = 10

    x, y = make_moons(n_samples=170, noise=0.1, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    base_model = EnsembleMLP(
        in_features=2,
        out_features=2,
        hidden_features=[128, 128],
        bias=True,
        norm="batch",
        dropout=0.1,
        embedding=EMBEDDING,
        embedding_num_items=100,
        attention=ATTENTION,
        attention_dropout=0.0,
        attention_inner_dim=ATTENTION_INNER_DIM,
        attention_num_heads=ATTENTION_NUM_HEADS,
        task="regression",
        num_members=5,
    )

    model = Ensemble(
        model=base_model,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=20,
        batch_size=32,
        seed=45,
        gpu=None,
        task="regression",
        logger=True,
        regression_dropout=0.3,
    )

    if EMBEDDING:
        x_train_data = quantize(x_train, BINS, min_value=-2.5, max_value=2.5)
        x_val_data = quantize(x_val, BINS, min_value=-2.5, max_value=2.5)
        x_test_data = quantize(x_test, BINS, min_value=-2.5, max_value=2.5)
    else:
        x_train_data = x_train
        x_val_data = x_val
        x_test_data = x_test

    print("x_train_data.shape", x_train_data.shape)
    print("y_train.shape", y_train.shape)

    model.fit(x_train_data, y_train, x_val_data, y_val)

    base_model.replace_output_layer(out_features=1)
    base_model.set_task("classification")
    base_model.reset_model()

    model = Ensemble(
        model=base_model,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=20,
        batch_size=32,
        seed=42,
        gpu=None,
        task="classification",
        logger=True,
    )

    model.fit(x_train_data, y_train, x_val_data, y_val)

    y_pred = model.predict(x_test_data)
    y_pred_proba = model.predict_proba(x_test_data, cache=True)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log loss: {log_loss(y_test, y_pred_proba):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, y_pred_proba):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("Probabilities of first 10 test samples:", y_pred_proba[:10])
    print("Predicted labels of first 10 test samples:", y_pred[:10])
    print("Ground truth labels of first 10 test samples:", y_test[:10])

    model.save("model.pt")
    cache = model.get_cache()
    cache[CACHE_TARGET_KEY] = y_test
    with open("cache.pt", "wb") as f:
        pkl.dump(cache, f)
    model.reset_cache()

    model.load("model.pt")
    os.remove("model.pt")

    x1 = np.linspace(-2.5, 2.5, 100)
    x2 = np.linspace(-2.5, 2.5, 100)
    xx1, xx2 = np.meshgrid(x1, x2)
    x_grid = np.stack([xx1, xx2], axis=2).reshape(-1, 2)
    if EMBEDDING:
        x_grid = quantize(x_grid, BINS, min_value=-2.5, max_value=2.5)

    y_grid_proba = model.predict_proba(x_grid)
    plt.contourf(xx1, xx2, y_grid_proba.reshape(xx1.shape), levels=100)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="bwr")
    plt.savefig("decision_boundary.png")
