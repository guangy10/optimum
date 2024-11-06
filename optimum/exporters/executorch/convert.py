"""ExecuTorch model check and export functions."""

import os
from pathlib import Path
from typing import Union

from transformers.utils import is_torch_available

from ...utils import (
    TORCH_MINIMUM_VERSION,
    check_if_transformers_greater,
    is_diffusers_available,
    logging,
)
from ..tasks import TasksManager
from .recipe_registry import recipe_registry


if is_torch_available():
    import torch
    from transformers.modeling_utils import PreTrainedModel


def export_to_executorch(
    model: Union["PreTrainedModel", "TorchExportableModuleWithStaticCache"],
    task: str,
    recipe: str,
    output_dir: Union[str, Path],
    **kwargs,
):
    """
    Full-suite ExecuTorch export function, exporting **from a pre-trained PyTorch model**. This function is especially useful in case one needs to do modifications on the model, as overriding a forward call, before exporting to ExecuTorch.

    Args:
        model (`Union["PreTrainedModel", "TorchExportableModuleWithStaticCache"]`):
            PyTorch model to export to ExecuTorch.
        recipe (`str`, defaults to `"xnnpack"`):
            The recipe to use to do the export. Defaults to "xnnpack".
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated ExecuTorch model.
    """
    try:
        recipe_func = recipe_registry.get(recipe)
    except KeyError as e:
        raise RuntimeError(f"The recipe '{recipe}' isn't registered. Detailed error: {e}")

    executorch_prog = recipe_func(model, task, kwargs)

    full_path = os.path.join(f"{output_dir}", "model.pte")
    with open(full_path, "wb") as f:
        executorch_prog.write_to_file(f)
        print(f"Saved exported program to {full_path}")

    return executorch_prog
