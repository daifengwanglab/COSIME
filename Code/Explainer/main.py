from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data_loader import load_data
from shapley_computation import (
    monte_carlo_shapley_early_fusion,
    monte_carlo_shapley_late_fusion,
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(log_file: str) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("shapley_cli")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def import_model_class(model_script_path: str):
    path = Path(model_script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model architecture script not found: {path}")

    module_name = f"shapley_user_model_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model architecture from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(f"{path} must define a class named Model.")
    return module.Model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise TypeError(
        "The checkpoint is not a recognizable state_dict. Expected a tensor dictionary, "
        "or a dictionary containing 'model_state_dict' or 'state_dict'."
    )


def load_model(
    model_path: str,
    model_format: str,
    model_script_path: str | None,
    input_dims: Sequence[int],
    dim: int,
    dropout: float,
    device: torch.device,
    strict: bool,
) -> torch.nn.Module:
    model_path = str(Path(model_path).expanduser().resolve())

    if model_format == "torchscript":
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model

    if model_script_path is None:
        raise ValueError(
            "--model_script is required for a state_dict checkpoint because the weights "
            "do not contain the Model architecture. Use the actual model.py from training; "
            "it does not have to be named user_model.py."
        )

    model_class = import_model_class(model_script_path)
    model = model_class(*input_dims, dim=dim, dropout=dropout).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"WARNING: missing checkpoint keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"WARNING: unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.eval()
    return model


def _as_prediction_tensors(output: Any, batch_size: int) -> list[torch.Tensor]:
    
    if torch.is_tensor(output):
        return [output.reshape(batch_size, -1).mean(dim=1)]

    if isinstance(output, (tuple, list)) and len(output) == 2:
        predictions = output[1]
        if torch.is_tensor(predictions):
            return [predictions.reshape(batch_size, -1).mean(dim=1)]
        if isinstance(predictions, (tuple, list)) and predictions and all(
            torch.is_tensor(item) for item in predictions
        ):
            return [item.reshape(batch_size, -1).mean(dim=1) for item in predictions]

    if isinstance(output, (tuple, list)) and output and all(
        torch.is_tensor(item) for item in output
    ):
        return [item.reshape(batch_size, -1).mean(dim=1) for item in output]

    raise TypeError(
        "Could not identify prediction tensor(s) in the model output. "
        f"Received output type: {type(output).__name__}."
    )


class CommandModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_dims: Sequence[int], fusion: str):
        super().__init__()
        self.model = model
        self.input_dims = tuple(int(value) for value in input_dims)
        self.fusion = fusion

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[1] != sum(self.input_dims):
            raise ValueError(
                f"X has {X.shape[1]} features, but --input_dims sums to {sum(self.input_dims)}."
            )
        modality_inputs = torch.split(X, self.input_dims, dim=1)
        output = self.model(*modality_inputs)
        predictions = _as_prediction_tensors(output, X.shape[0])

        if self.fusion == "early":
            if len(predictions) != 1:
                raise ValueError(
                    "--fusion early requires one final prediction tensor, but the model returned "
                    f"{len(predictions)} modality-specific prediction tensors. This architecture "
                    "is a late-fusion model; run with --fusion late or use the correct early-fusion model."
                )
            return predictions[0]

        return torch.stack(predictions, dim=0).mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Monte Carlo Shapley values and interactions")
    parser.add_argument("--input_data", required=True, help="Numeric model-input CSV")
    parser.add_argument("--input_model", required=True, help="Trained checkpoint or TorchScript model")
    parser.add_argument("--model_format", choices=["state_dict", "torchscript"], default="state_dict")
    parser.add_argument(
        "--model_script",
        help="Python file defining class Model; required only for --model_format state_dict",
    )
    parser.add_argument("--input_dims", required=True, help="Comma-separated modality dimensions, e.g. 100,100")
    parser.add_argument("--fusion", choices=["early", "late"], required=True)
    parser.add_argument("--save", required=True, help="Output directory")
    parser.add_argument("--log", help="Log file; defaults to <save>/shapley.log")
    parser.add_argument("--drop_columns", default="", help="Comma-separated non-feature CSV columns to remove")
    parser.add_argument("--expected_samples", type=int, help="Fail unless this many input rows are loaded")
    parser.add_argument("--dim", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mc_iterations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--interaction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_memory_usage_gb", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--allow_non_strict",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys; strict loading is safer and is the default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.save).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log or str(output_dir / "shapley.log")
    logger = setup_logger(log_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dims = tuple(int(value.strip()) for value in args.input_dims.split(",") if value.strip())
    if not input_dims or any(value < 1 for value in input_dims):
        raise ValueError("--input_dims must contain positive integers, e.g. 100,100")

    drop_columns = [value.strip() for value in args.drop_columns.split(",") if value.strip()]
    X, feature_names = load_data(
        args.input_data,
        drop_columns=drop_columns,
        expected_samples=args.expected_samples,
    )
    if X.shape[1] != sum(input_dims):
        raise ValueError(
            f"Loaded X has {X.shape[1]} features, but --input_dims sums to {sum(input_dims)}."
        )

    device = choose_device(args.device)
    print(f"Computation device: {device}")
    X = X.to(device)

    base_model = load_model(
        model_path=args.input_model,
        model_format=args.model_format,
        model_script_path=args.model_script,
        input_dims=input_dims,
        dim=args.dim,
        dropout=args.dropout,
        device=device,
        strict=not args.allow_non_strict,
    )
    model = CommandModelWrapper(base_model, input_dims=input_dims, fusion=args.fusion).to(device)
    model.eval()

    function = (
        monte_carlo_shapley_early_fusion
        if args.fusion == "early"
        else monte_carlo_shapley_late_fusion
    )
    function(
        model=model,
        X=X,
        mc_iterations=args.mc_iterations,
        batch_size=args.batch_size,
        interaction=args.interaction,
        max_memory_usage_gb=args.max_memory_usage_gb,
        logger=logger,
        export_dir=str(output_dir),
        seed=args.seed,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    main()
