"""ExecuTorchModelForXXX classes, allowing to run ExecuTorch Models with ExecuTorch Runtime using the same API as Transformers."""

import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationMixin,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    CausalLMOutputWithPast,
    ModelOutput,
)

from ..exporters import TasksManager
from ..exporters.executorch import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel

if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class ExecuTorchModelForCausalLM(OptimizedModel):
    """
    ExecuTorch model with a causal language modeling head for ExecuTorch Runtime inference.
    """

    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        model: "ExecuTorchModule",  # Similar to ort.InferenceSession for ORTModel
        config: "PretrainedConfig",
    ):
        super().__init__(model, config)
        # TODO: Remove once figure out how make `ExecuTorchModule` inherit from `PreTrainedModel`
        self.et_model = model

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        return self.et_model.forward(input_ids, cache_position)[0]

    @classmethod
    def _from_pretrained(
        cls,
        model_dir_path: Union[str, Path],
        task: str,
        recipe: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
    ) -> "ExecuTorchModelForCausalLM":
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        full_path = os.path.join(f"{model_dir_path}", "model.pte")
        model = _load_for_executorch(full_path)

        return cls(
            model=model,
            config=config,
        )

    @classmethod
    def _export(
        cls,
        model_id: str,
        task: str,
        recipe: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export to ExecuTorch and save the pte file to the temporary directory
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            recipe=recipe,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        return cls._from_pretrained(
            model_dir_path=save_dir_path,
            task=task,
            recipe=recipe,
            config=config,
            use_auth_token=use_auth_token,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

    def generate(
        self,
        prompt_tokens: List[int],
    ) -> List[int]:
        """
        Generate a sequence of token ids using the ExecuTorchModule.

        `pipeline()` is where everything puts together. It consists of the tokenizer for encoding the inputs and decoding the model generated outputs.
        """
        pass
