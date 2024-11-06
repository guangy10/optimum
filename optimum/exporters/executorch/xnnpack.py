from typing import Union

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
import torch
import torch.export._trace

from .recipe_registry import register_recipe


@register_recipe("xnnpack")
def export_to_executorch_with_xnnpack(
    model,
    task: str,
    **kwargs,
):
    if task == "text-generation":
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
    else:
        # TODO: Prepare model inputs for other tasks
        raise ValueError(f"Unsupported task {task}.")

    with torch.no_grad():
        exported_program = torch.export._trace._export(
            model,
            args=(example_input_ids,),
            kwargs={"cache_position": example_cache_position},
            pre_dispatch=False,
            strict=True,
        )

        return to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _skip_dim_order=True,
            ),
        ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))
