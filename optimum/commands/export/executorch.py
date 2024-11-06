"""Defines the command line for the export with ExecuTorch."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_executorch(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store the generated ExecuTorch model."
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
        ),
    )
    optional_group.add_argument(
        "--recipe",
        type=str,
        default="xnnpack_fp32",
        help='Pre-defined recipes for export to ExecuTorch. Defaults to "xnnpack_fp32".',
    )


class ExecuTorchExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_executorch(parser)

    def run(self):
        from ...exporters.executorch import main_export

        main_export(
            model_name_or_path=self.args.model,
            output=self.args.output,
            task=self.args.task,
            recipe=self.args.recipe,
        )
