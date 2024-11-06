"""Entry point to the optimum.exporters.executorch command line."""

import argparse
import warnings
from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import is_torch_available

from ...commands.export.executorch import parse_args_executorch
from ...utils import logging
from ..tasks import TasksManager
from .convert import export_to_executorch
from .task_registry import task_registry

if is_torch_available():
    import torch

from typing import Optional, Union


logger = logging.get_logger()


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str,
    recipe: str = "xnnpack",
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    **kwargs,
):
    """
    Full-suite ExecuTorch export function, exporting **from a model ID on Hugging Face Hub or a local model repository**.

    Args:
        > Required parameters

        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export. Example: `model_name_or_path="google/gemma-2b"` or `mode_name_or_path="/path/to/model_folder`.
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated ExecuTorch model.

        > Optional parameters

        task (`Optional[str]`, defaults to `None`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model. For decoder models,
            use `xxx-with-past` to export the model using past key values in the decoder.
        recipe (`str`, defaults to `"xnnpack"`):
            The recipe to use to do the export. Defaults to "xnnpack".
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        pad_token_id (`Optional[int]`, defaults to `None`):
            This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        force_download (`bool`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
            Deprecated. Please use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

    Example usage:
    ```python
    >>> from optimum.exporters.executorch import main_export

    >>> main_export("gemma-2b", output="gemma-2b_onnx/")
    ```
    """

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    model = task_registry.get(task)(model_name_or_path, kwargs)

    if task == "text-generation":
        from transformers.integrations.executorch import TorchExportableModuleWithStaticCache

        model = TorchExportableModuleWithStaticCache(model)

    return export_to_executorch(
        model=model,
        task=task,
        recipe=recipe,
        output=output,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser("Hugging Face Optimum ExecuTorch exporter")

    parse_args_executorch(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        task=args.task,
        recipe=args.recipe,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=args.pad_token_id,
    )


if __name__ == "__main__":
    main()
