# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Dict, TypeVar

from torch.optim import Optimizer

from fairseq2.checkpoint import CheckpointManager, OutOfProcHuggingFaceSaver
from fairseq2.context import RuntimeContext
from fairseq2.datasets import DataReader
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics.recorders import MetricDescriptor, UnknownMetricDescriptorError
from fairseq2.optim.lr_scheduler import LRScheduler
from fairseq2.recipes import EvalUnit, Model, Trainer, TrainUnit, Validator
from fairseq2.recipes.config import CommonSection, RegimeSection, TrainerSection
from fairseq2.utils.gc import (
    CPythonGarbageCollector,
    GarbageCollector,
    NoopGarbageCollector,
)
from fairseq2.utils.threading import get_default_thread_pool

# isort: split

from fairseq2.recipes.common._device import create_device_stat_tracker
from fairseq2.recipes.common._error import HuggingFaceNotSupportedError
from fairseq2.recipes.common._metrics import create_metric_recorder
from fairseq2.recipes.common._profilers import create_profiler

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


def create_trainer(
    context: RuntimeContext,
    trainer_section: TrainerSection,
    regime_section: RegimeSection,
    common_section: CommonSection,
    output_dir: Path,
    unit: TrainUnit[BatchT],
    data_readers: Sequence[DataReader[BatchT]],
    valid_units: Sequence[EvalUnit[BatchT]],
    valid_data_readers: Sequence[DataReader[BatchT]],
    gangs: Gangs,
    checkpoint_manager_attacker: CheckpointManager,
    checkpoint_manager_defender: CheckpointManager,
    checkpoint_manager_ema_attacker: CheckpointManager,
    checkpoint_manager_ema_defender: CheckpointManager,
    checkpoint_manager_ref_attacker: CheckpointManager,
    checkpoint_manager_ref_defender: CheckpointManager,
    optimizer_attacker: Optimizer,
    optimizer_defender: Optimizer,
    lr_scheduler_attacker: LRScheduler,
    lr_scheduler_defender: LRScheduler,
    seed: int,
    *,
    hyper_params: object = None,
    score_metric: str | None = None,
) -> Trainer:
    if regime_section.save_as_hugging_face:
        if not unit.model_attacker.handler.supports_hugging_face:
            raise HuggingFaceNotSupportedError(unit.model_attacker.name)
        if not unit.model_defender.handler.supports_hugging_face:
            raise HuggingFaceNotSupportedError(unit.model_defender.name)

        checkpoint_dir_attacker = output_dir.joinpath("attacker/checkpoints")
        checkpoint_dir_defender = output_dir.joinpath("defender/checkpoints")
        checkpoint_dir_ema_attacker = output_dir.joinpath("ema_attacker/checkpoints")
        checkpoint_dir_ema_defender = output_dir.joinpath("ema_defender/checkpoints")
        checkpoint_dir_ref_attacker = output_dir.joinpath("ref_attacker/checkpoints")
        checkpoint_dir_ref_defender = output_dir.joinpath("ref_defender/checkpoints")

        thread_pool = get_default_thread_pool()

        hugging_face_saver_attacker = OutOfProcHuggingFaceSaver(
            checkpoint_dir_attacker, thread_pool
        )
        hugging_face_saver_defender = OutOfProcHuggingFaceSaver(
            checkpoint_dir_defender, thread_pool
        )
        hugging_face_saver_ema_attacker = OutOfProcHuggingFaceSaver(
            checkpoint_dir_ema_attacker, thread_pool
        )
        hugging_face_saver_ema_defender = OutOfProcHuggingFaceSaver(
            checkpoint_dir_ema_defender, thread_pool
        )
        hugging_face_saver_ref_attacker = OutOfProcHuggingFaceSaver(
            checkpoint_dir_ref_attacker, thread_pool
        )
        hugging_face_saver_ref_defender = OutOfProcHuggingFaceSaver(
            checkpoint_dir_ref_defender, thread_pool
        )
    else:
        hugging_face_saver_attacker = None
        hugging_face_saver_defender = None
        hugging_face_saver_ema_attacker = None
        hugging_face_saver_ema_defender = None
        hugging_face_saver_ref_attacker = None
        hugging_face_saver_ref_defender = None

    score_metric_descriptor = _get_score_metric_descriptor(context, score_metric)

    metric_recorder = create_metric_recorder(
        context, common_section, gangs, output_dir, hyper_params
    )

    profiler = create_profiler(context, common_section, gangs, output_dir)

    garbage_collector = _create_garbage_collector(context, trainer_section)

    device_stat_tracker = create_device_stat_tracker(gangs)

    # TODO: Fix once we support static mixed precision on single device.
    if trainer_section.mixed_precision == "static":
        amp = gangs.root.size == 1 or trainer_section.data_parallelism != "fsdp"
    else:
        amp = trainer_section.mixed_precision == "dynamic"

    if gangs.root.device.type == "cpu":
        log.warning("Based on your environment setup the training will be run on CPU. If this was not intended, check your job options (e.g. pass `--gpus-per-node` on Slurm).")  # fmt: skip

    if valid_units:
        validator = Validator(
            units=valid_units,
            data_readers=valid_data_readers,
            gangs=gangs,
            dtype=trainer_section.dtype,
            amp=amp,
            score_metric_descriptor=score_metric_descriptor,
            checkpoint_manager_attacker=checkpoint_manager_attacker,
            checkpoint_manager_defender=checkpoint_manager_defender,
            seed=seed,
            metric_recorder=metric_recorder,
            profiler=profiler,
            device_stat_tracker=device_stat_tracker,
            wall_watch=context.wall_watch,
            progress_reporter=context.progress_reporter,
        )
    else:
        validator = None

    # fmt: off
    return Trainer(
        unit=unit,
        data_readers=data_readers,
        gangs=gangs,
        dtype=trainer_section.dtype,
        amp=amp,
        optimizer_attacker=optimizer_attacker,
        optimizer_defender=optimizer_defender,
        lr_scheduler_attacker=lr_scheduler_attacker,
        lr_scheduler_defender=lr_scheduler_defender,
        fp16_loss_scale=trainer_section.fp16_loss_scale,
        no_sync_grad_accumulation=trainer_section.grad_accumulation.no_sync,
        max_grad_norm=trainer_section.max_grad_norm,
        grad_check=trainer_section.grad_check,
        anomaly_detection=trainer_section.anomaly_detection,
        train_attacker=trainer_section.train_attacker,
        train_defender=trainer_section.train_defender,
        ema_attacker_ema_decay=trainer_section.ema_attacker_ema_decay,
        ema_defender_ema_decay=trainer_section.ema_defender_ema_decay,
        ref_attacker_ema_decay=trainer_section.ref_attacker_ema_decay,
        ref_defender_ema_decay=trainer_section.ref_defender_ema_decay,
        summon_full_parameters_for_validation=trainer_section.summon_full_parameters_for_validation,
        seed=seed,
        max_num_steps=regime_section.num_steps,
        max_num_data_epochs=regime_section.num_data_epochs,
        validator=validator,
        validate_at_start=regime_section.validate_at_start,
        validate_after_n_steps=regime_section.validate_after_n_steps,
        validate_every_n_steps=regime_section.validate_every_n_steps,
        validate_after_n_data_epochs=regime_section.validate_after_n_data_epochs,
        validate_every_n_data_epochs=regime_section.validate_every_n_data_epochs,
        score_metric_descriptor=score_metric_descriptor,
        checkpoint_manager_attacker=checkpoint_manager_attacker,
        checkpoint_manager_defender=checkpoint_manager_defender,
        checkpoint_manager_ema_attacker=checkpoint_manager_ema_attacker,
        checkpoint_manager_ema_defender=checkpoint_manager_ema_defender,
        checkpoint_manager_ref_attacker=checkpoint_manager_ref_attacker,
        checkpoint_manager_ref_defender=checkpoint_manager_ref_defender,
        checkpoint_after_n_steps=regime_section.checkpoint_after_n_steps,
        checkpoint_every_n_steps=regime_section.checkpoint_every_n_steps,
        checkpoint_after_n_data_epochs=regime_section.checkpoint_after_n_data_epochs,
        checkpoint_every_n_data_epochs=regime_section.checkpoint_every_n_data_epochs,
        checkpoint_models=regime_section.checkpoint_models,
        checkpoint_ema_models=regime_section.checkpoint_ema_models,
        checkpoint_ref_models=regime_section.checkpoint_ref_models,
        save_model_only=regime_section.save_model_only,
        hugging_face_saver_attacker=hugging_face_saver_attacker,
        hugging_face_saver_defender=hugging_face_saver_defender,
        hugging_face_saver_ema_attacker=hugging_face_saver_ema_attacker,
        hugging_face_saver_ema_defender=hugging_face_saver_ema_defender,
        hugging_face_saver_ref_attacker=hugging_face_saver_ref_attacker,
        hugging_face_saver_ref_defender=hugging_face_saver_ref_defender,
        keep_last_n_checkpoints=regime_section.keep_last_n_checkpoints,
        keep_best_n_checkpoints=regime_section.keep_best_n_checkpoints,
        keep_checkpoint_every_n_steps=regime_section.keep_checkpoint_every_n_steps,
        metric_recorder=metric_recorder,
        publish_metrics_after_n_steps=regime_section.publish_metrics_after_n_steps,
        publish_metrics_every_n_steps=regime_section.publish_metrics_every_n_steps,
        publish_metrics_after_n_data_epochs=regime_section.publish_metrics_after_n_data_epochs,
        publish_metrics_every_n_data_epochs=regime_section.publish_metrics_every_n_data_epochs,
        garbage_collector=garbage_collector,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=context.wall_watch,
        progress_reporter=context.progress_reporter,
    )
    # fmt: on


def _get_score_metric_descriptor(
    context: RuntimeContext, score_metric: str | None
) -> MetricDescriptor | None:
    if score_metric is None:
        return None

    metric_descriptors = context.get_registry(MetricDescriptor)

    try:
        return metric_descriptors.get(score_metric)
    except LookupError:
        raise UnknownMetricDescriptorError(score_metric) from None


def _create_garbage_collector(
    context: RuntimeContext, trainer_section: TrainerSection
) -> GarbageCollector:
    if trainer_section.gc_every_n_steps is None:
        return NoopGarbageCollector()

    return CPythonGarbageCollector(trainer_section.gc_every_n_steps)
