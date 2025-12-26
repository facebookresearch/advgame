# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from enum import Enum
from typing import List, cast

import ray
import torch
import torch.nn as nn
from torch import Tensor
from vllm import RequestOutput

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    SequenceData,
)
from fairseq2.datasets import (
    SequenceBatch,
)
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.nn._batch_layout import BatchLayout
from fairseq2.recipes.lm._online_finetune._remote_model import RemoteVllmModel
from fairseq2.utils.list import lget_shape, lslice


class xError(Enum):
    MAX_LENGTH = 1
    PARSING = 2
    BAD_FAITH_JUDGE = 3
    NOT_FAITHFUL = 4
    NO_VALID_RESPONSE = 5
    BAD_ATT_JUDGE = 6
    NO_ERROR = 7


class yError(Enum):
    PROMPT_NOT_VALID = 1
    MAX_LENGTH = 2
    PARSING = 3
    BAD_DEF_JUDGE = 4
    NO_ERROR = 5


@dataclass
class RolloutTree:
    s_category_B: List[str]
    s_text_B: List[str]
    s_emb_text_B: List[str]
    s_emb_tokens_B: List[List[int]]
    x_rollout_BA: List[List[RequestOutput | None]]
    x_raw_text_BA: List[List[str | None]]
    x_raw_tokens_BA: List[List[List[int] | None]]
    x_text_BA: List[List[str | None]]
    x_tokens_BA: List[List[List[int] | None]]
    x_cot_text_BA: List[List[str | None]]
    x_cot_tokens_BA: List[List[List[int] | None]]
    x_emb_text_BA: List[List[str | None]]
    x_emb_tokens_BA: List[List[List[int] | None]]
    x_error_BA: List[List[xError]]
    y_rollout_BAD: List[List[List[RequestOutput | None]]]
    y_raw_text_BAD: List[List[List[str | None]]]
    y_raw_tokens_BAD: List[List[List[List[int] | None]]]
    y_text_BAD: List[List[List[str | None]]]
    y_tokens_BAD: List[List[List[List[int] | None]]]
    y_cot_text_BAD: List[List[List[str | None]]]
    y_cot_tokens_BAD: List[List[List[List[int] | None]]]
    y_error_BAD: List[List[List[yError]]]
    B: int
    A: int
    D: int

    def __post_init__(self):
        B, A, D = self.B, self.A, self.D
        assert (B,) == lget_shape(self.s_category_B, depth=1)
        assert (B,) == lget_shape(self.s_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_tokens_B, depth=1)
        assert (B, A) == lget_shape(self.x_rollout_BA, depth=2)
        assert (B, A) == lget_shape(self.x_raw_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_raw_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_emb_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_emb_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_error_BA, depth=2)
        assert (B, A, D) == lget_shape(self.y_rollout_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_raw_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_raw_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_cot_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_cot_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_error_BAD, depth=3)


@dataclass
class ScoredRolloutTree:
    s_category_B: List[str]
    s_text_B: List[str]
    s_emb_text_B: List[str]
    s_emb_tokens_B: List[List[int]]
    x_rollout_BA: List[List[RequestOutput | None]]
    x_raw_text_BA: List[List[str | None]]
    x_raw_tokens_BA: List[List[List[int] | None]]
    x_text_BA: List[List[str | None]]
    x_tokens_BA: List[List[List[int] | None]]
    x_cot_text_BA: List[List[str | None]]
    x_cot_tokens_BA: List[List[List[int] | None]]
    x_emb_text_BA: List[List[str | None]]
    x_emb_tokens_BA: List[List[List[int] | None]]
    x_error_BA: List[List[xError]]
    x_reward_BA: List[List[float | None]]
    x_reward_patched_BA: List[List[float | None]]
    x_advantage_BA: List[List[float | None]]
    y_rollout_BAD: List[List[List[RequestOutput | None]]]
    y_raw_text_BAD: List[List[List[str | None]]]
    y_raw_tokens_BAD: List[List[List[List[int] | None]]]
    y_text_BAD: List[List[List[str | None]]]
    y_tokens_BAD: List[List[List[List[int] | None]]]
    y_cot_text_BAD: List[List[List[str | None]]]
    y_cot_tokens_BAD: List[List[List[List[int] | None]]]
    y_error_BAD: List[List[List[yError]]]
    y_reward_BAD: List[List[List[float | None]]]
    y_reward_patched_BAD: List[List[List[float | None]]]
    y_advantage_BAD: List[List[List[float | None]]]
    B: int
    A: int
    D: int

    def __post_init__(self):
        B, A, D = self.B, self.A, self.D
        assert (B,) == lget_shape(self.s_category_B, depth=1)
        assert (B,) == lget_shape(self.s_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_tokens_B, depth=1)
        assert (B, A) == lget_shape(self.x_rollout_BA, depth=2)
        assert (B, A) == lget_shape(self.x_raw_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_raw_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_emb_text_BA, depth=2)
        assert (B, A) == lget_shape(self.x_emb_tokens_BA, depth=2)
        assert (B, A) == lget_shape(self.x_error_BA, depth=2)
        assert (B, A) == lget_shape(self.x_reward_BA, depth=2)
        assert (B, A) == lget_shape(self.x_reward_patched_BA, depth=2)
        assert (B, A) == lget_shape(self.x_advantage_BA, depth=2)
        assert (B, A, D) == lget_shape(self.y_rollout_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_raw_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_raw_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_cot_text_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_cot_tokens_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_error_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_reward_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_reward_patched_BAD, depth=3)
        assert (B, A, D) == lget_shape(self.y_advantage_BAD, depth=3)


@dataclass
class RolloutTreeDirect:
    s_category_B: List[str]
    s_text_B: List[str]
    s_emb_text_B: List[str]
    s_emb_tokens_B: List[List[int]]
    y_rollout_BD: List[List[RequestOutput | None]]
    y_raw_text_BD: List[List[str | None]]
    y_raw_tokens_BD: List[List[List[int] | None]]
    y_text_BD: List[List[str | None]]
    y_tokens_BD: List[List[List[int] | None]]
    y_cot_text_BD: List[List[str | None]]
    y_cot_tokens_BD: List[List[List[int] | None]]
    y_error_BD: List[List[bool]]
    B: int
    D: int

    def __post_init__(self):
        B, D = self.B, self.D
        assert (B,) == lget_shape(self.s_category_B, depth=1)
        assert (B,) == lget_shape(self.s_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_tokens_B, depth=1)
        assert (B, D) == lget_shape(self.y_rollout_BD, depth=2)
        assert (B, D) == lget_shape(self.y_raw_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_raw_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_cot_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_cot_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_error_BD, depth=2)


@dataclass
class ScoredRolloutTreeDirect:
    s_category_B: List[str]
    s_text_B: List[str]
    s_emb_text_B: List[str]
    s_emb_tokens_B: List[List[int]]
    y_rollout_BD: List[List[RequestOutput | None]]
    y_raw_text_BD: List[List[str | None]]
    y_raw_tokens_BD: List[List[List[int] | None]]
    y_text_BD: List[List[str | None]]
    y_tokens_BD: List[List[List[int] | None]]
    y_cot_text_BD: List[List[str | None]]
    y_cot_tokens_BD: List[List[List[int] | None]]
    y_error_BD: List[List[bool]]
    y_reward_BD: List[List[float | None]]
    y_reward_patched_BD: List[List[float | None]]
    y_advantage_BD: List[List[float | None]]
    B: int
    D: int

    def __post_init__(self):
        B, D = self.B, self.D
        assert (B,) == lget_shape(self.s_category_B, depth=1)
        assert (B,) == lget_shape(self.s_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_text_B, depth=1)
        assert (B,) == lget_shape(self.s_emb_tokens_B, depth=1)
        assert (B, D) == lget_shape(self.y_rollout_BD, depth=2)
        assert (B, D) == lget_shape(self.y_raw_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_raw_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_cot_text_BD, depth=2)
        assert (B, D) == lget_shape(self.y_cot_tokens_BD, depth=2)
        assert (B, D) == lget_shape(self.y_error_BD, depth=2)
        assert (B, D) == lget_shape(self.y_reward_BD, depth=2)
        assert (B, D) == lget_shape(self.y_reward_patched_BD, depth=2)
        assert (B, D) == lget_shape(self.y_advantage_BD, depth=2)


class StatefulRolloutBagGame:
    bag_step_b: int = 0
    bag_step_a: int = 0
    bag_step_d: int = 0
    total_bag_step: int = 0
    _trainer_step: int = None

    def __init__(
        self, num_microbatches_B: int, num_microbatches_A: int, num_microbatches_D: int
    ):
        self.srt: ScoredRolloutTree | None = None
        self.num_microbatches_B = num_microbatches_B
        self.num_microbatches_A = num_microbatches_A
        self.num_microbatches_D = num_microbatches_D
        self.total_max_bag_steps = (
            self.num_microbatches_B * self.num_microbatches_A * self.num_microbatches_D
        )

    def maybe_reset_bag(self, trainer_step: int):
        # this is called every train step to see if we need to reset the bag
        if self.total_bag_step >= self.total_max_bag_steps:
            log.info(f"Resetting rollout bag at trainer step {trainer_step}")
            # new trainer step, reset bag and counters
            self.srt = None
            self.bag_step_b = 0
            self.bag_step_a = 0
            self.bag_step_d = 0
            self.total_bag_step = 0
            self._trainer_step = trainer_step

    def save(self, srt: ScoredRolloutTree | ScoredRolloutTreeDirect):
        if isinstance(srt, ScoredRolloutTreeDirect):
            raise NotImplementedError(
                "StatefulRolloutBagGame only supports ScoredRolloutTree for now."
            )
        self.srt = srt
        B, A, D = srt.B, srt.A, srt.D
        assert B % self.num_microbatches_B == 0, f"{B} {self.num_microbatches_B}"
        assert A % self.num_microbatches_A == 0, f"{A} {self.num_microbatches_A}"
        assert D % self.num_microbatches_D == 0, f"{D} {self.num_microbatches_D}"

    def load(self):
        if (
            self.num_microbatches_B == 1
            and self.num_microbatches_A == 1
            and self.num_microbatches_D == 1
        ):
            self.total_bag_step += 1
            return self.srt

        srt = self.srt

        b = srt.B // self.num_microbatches_B
        a = srt.A // self.num_microbatches_A
        d = srt.D // self.num_microbatches_D

        b1 = self.bag_step_b * b
        b2 = b1 + b
        a1 = self.bag_step_a * a
        a2 = a1 + a
        d1 = self.bag_step_d * d
        d2 = d1 + d

        log.info(
            f"[RolloutBag] total_bag_step={self.total_bag_step + 1}/{self.total_max_bag_steps}, bag_step=({self.bag_step_b + 1}/{self.num_microbatches_B}, {self.bag_step_a + 1}/{self.num_microbatches_A}, {self.bag_step_d + 1}/{self.num_microbatches_D}), slice=({b1}:{b2}, {a1}:{a2}, {d1}:{d2})]"
        )

        def slice_b(x):
            return lslice(x, b1, b2, axis=0)

        def slice_ba(x):
            return lslice(slice_b(x), a1, a2, axis=1)

        def slice_bad(x):
            return lslice(slice_ba(x), d1, d2, axis=2)

        s_category_b = slice_b(srt.s_category_B)
        s_text_b = slice_b(srt.s_text_B)
        s_emb_text_b = slice_b(srt.s_emb_text_B)
        s_emb_tokens_b = slice_b(srt.s_emb_tokens_B)

        x_rollout_ba = slice_ba(srt.x_rollout_BA)
        x_raw_text_ba = slice_ba(srt.x_raw_text_BA)
        x_raw_tokens_ba = slice_ba(srt.x_raw_tokens_BA)
        x_text_ba = slice_ba(srt.x_text_BA)
        x_tokens_ba = slice_ba(srt.x_tokens_BA)
        x_cot_text_ba = slice_ba(srt.x_cot_text_BA)
        x_cot_tokens_ba = slice_ba(srt.x_cot_tokens_BA)
        x_emb_text_ba = slice_ba(srt.x_emb_text_BA)
        x_emb_tokens_ba = slice_ba(srt.x_emb_tokens_BA)
        x_error_ba = slice_ba(srt.x_error_BA)
        x_reward_ba = slice_ba(srt.x_reward_BA)
        x_reward_patched_ba = slice_ba(srt.x_reward_patched_BA)
        x_advantage_ba = slice_ba(srt.x_advantage_BA)

        y_rollout_bad = slice_bad(srt.y_rollout_BAD)
        y_raw_text_bad = slice_bad(srt.y_raw_text_BAD)
        y_raw_tokens_bad = slice_bad(srt.y_raw_tokens_BAD)
        y_text_bad = slice_bad(srt.y_text_BAD)
        y_tokens_bad = slice_bad(srt.y_tokens_BAD)
        y_cot_text_bad = slice_bad(srt.y_cot_text_BAD)
        y_cot_tokens_bad = slice_bad(srt.y_cot_tokens_BAD)
        y_error_bad = slice_bad(srt.y_error_BAD)
        y_reward_bad = slice_bad(srt.y_reward_BAD)
        y_reward_patched_bad = slice_bad(srt.y_reward_patched_BAD)
        y_advantage_bad = slice_bad(srt.y_advantage_BAD)

        rollout_tree_slice = ScoredRolloutTree(
            s_category_B=s_category_b,
            s_text_B=s_text_b,
            s_emb_text_B=s_emb_text_b,
            s_emb_tokens_B=s_emb_tokens_b,
            x_rollout_BA=x_rollout_ba,
            x_raw_text_BA=x_raw_text_ba,
            x_raw_tokens_BA=x_raw_tokens_ba,
            x_text_BA=x_text_ba,
            x_tokens_BA=x_tokens_ba,
            x_cot_text_BA=x_cot_text_ba,
            x_cot_tokens_BA=x_cot_tokens_ba,
            x_emb_text_BA=x_emb_text_ba,
            x_emb_tokens_BA=x_emb_tokens_ba,
            x_error_BA=x_error_ba,
            x_reward_BA=x_reward_ba,
            x_reward_patched_BA=x_reward_patched_ba,
            x_advantage_BA=x_advantage_ba,
            y_rollout_BAD=y_rollout_bad,
            y_raw_text_BAD=y_raw_text_bad,
            y_raw_tokens_BAD=y_raw_tokens_bad,
            y_text_BAD=y_text_bad,
            y_tokens_BAD=y_tokens_bad,
            y_cot_text_BAD=y_cot_text_bad,
            y_cot_tokens_BAD=y_cot_tokens_bad,
            y_error_BAD=y_error_bad,
            y_reward_BAD=y_reward_bad,
            y_reward_patched_BAD=y_reward_patched_bad,
            y_advantage_BAD=y_advantage_bad,
            B=b,
            A=a,
            D=d,
        )

        if self.bag_step_d + 1 >= self.num_microbatches_D:
            self.bag_step_d = 0
            if self.bag_step_a + 1 >= self.num_microbatches_A:
                self.bag_step_a = 0
                self.bag_step_b += 1
            else:
                self.bag_step_a += 1
        else:
            self.bag_step_d += 1
        self.total_bag_step += 1
        return rollout_tree_slice


class StatefulRolloutBag:
    """A stateful container for managing and reusing model rollouts across multiple micro-batches.

    This class enables efficient gradient accumulation in GRPO by:
    1. Generating rollouts once per training step
    2. Reusing these rollouts across multiple forward passes (micro-batches)
    3. Managing the windowing of rollouts for each micro-batch

    In GRPO training, generating rollouts is computationally expensive. When the group_size
    is large (many rollouts per prompt), processing all rollouts in a single forward pass
    may exceed memory limits. This class allows splitting the computation into smaller
    chunks by tracking which subset of rollouts should be used in each forward pass.

    Usage in GRPO:
    - At the beginning of each training step, call `maybe_reset_bag(step_nr)`
    - If bag is empty (first micro-batch of step), generate rollouts and save them
    - For subsequent micro-batches, reuse the same rollouts
    - Use `get_rollout_start_end()` to determine which slice of rollouts to process
      in the current micro-batch based on forward_group_size

    Attributes:
        bag_step: Current micro-batch step within the training step
        _trainer_step: Current training step
        rollouts: List of model rollouts generated for the current step
        reward_outputs: List of reward outputs for the rollouts
    """

    bag_step: int = 0
    _trainer_step: int = None

    def __init__(self, max_bag_steps):
        self.rollouts: List = []
        self.reward_outputs: List = []
        self.max_bag_steps = max_bag_steps

    def maybe_reset_bag(self, trainer_step):
        # this is called every train step to see if we need to reset the bag
        if self.bag_step == self.max_bag_steps:
            # new trainer step, reset bag and counters
            self.rollouts = []
            self.reward_outputs = []
            self.bag_step = 0
            self._trainer_step = trainer_step

    def __len__(self):
        return len(self.rollouts)

    def save(self, rollouts, reward_outputs):
        self.rollouts = rollouts
        self.reward_outputs = reward_outputs
        self.bag_step += 1

    def load(self):
        self.bag_step += 1
        return self.rollouts, self.reward_outputs

    def get_rollout_start_end(self, num_rollout_per_forward: int):
        start_i = (self.bag_step - 1) * num_rollout_per_forward
        end_i = start_i + num_rollout_per_forward
        return start_i, end_i


@dataclass(kw_only=True)
class OnlineCriterionSection:
    name: str
    config: object


@dataclass(kw_only=True)
class VllmSyncSection:
    sync_model_every_n_steps: int = 1
    """How often to sync the vLLM model with the policy that is trained. -1 disables syncing."""

    sync_ref_model_every_n_steps: int = -1
    """How often to sync the reference vLLM model with the policy that is trained. -1 disables syncing."""


@dataclass(kw_only=True)
class VllmGameSyncSection:
    sync_model_every_n_steps: int = 1
    """How often to sync the vLLM model with the policy that is trained. -1 disables syncing."""

    sync_ema_model_instead_of_train_model: bool = False
    """Whether to sync the ema model instead of the training model to the VLLM workers."""


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def get_ray_actor(gangs: Gang, actor_name):
    # only retrieve vllm actors on main rank process as a safety measure to avoid blocking
    if gangs.dp.rank == 0 and gangs.tp.rank == 0:
        actor = ray.get_actor(actor_name)
    else:
        actor = None

    return actor


def rollout_to_vllm_logps(rollout: RequestOutput) -> Tensor:
    prompt_vllm_logps = [
        list(d.values())[0].logprob for d in rollout.prompt_logprobs[1:]
    ]
    assert len(rollout.outputs) == 1
    output_vllm_logps = [
        list(d.values())[0].logprob for d in rollout.outputs[0].logprobs
    ]
    return torch.tensor(prompt_vllm_logps + output_vllm_logps)


def _collate(sequences_B, prompt_lengths_B, device):
    sequences_B = [torch.tensor(sequence, device=device) for sequence in sequences_B]
    prompt_lengths_B = torch.tensor(prompt_lengths_B)
    sequences_B = collate_with_target_mask(sequences_B, prompt_lengths_B, device=device)
    return sequences_B


def collate_with_target_mask(
    list_of_tensors, prompt_lengths, pad_value=0, device="cpu"
):
    # list_of_tensors contain prompt+rollout tokens, we use prompt_len to define the target loss mask here
    to_collate = []
    for seq, prompt_len in zip(list_of_tensors, prompt_lengths):
        target_loss_mask = torch.arange(len(seq)) >= prompt_len
        to_collate.append({"seqs": seq, "target_loss_mask": target_loss_mask})

    target_mask_collate_opts = [
        CollateOptionsOverride("target_loss_mask", pad_value=False),
    ]
    collater = Collater(
        pad_value=pad_value, pad_to_multiple=1, overrides=target_mask_collate_opts
    )
    # from fairseq2.utils.env import get_rank
    # from os import environ
    # if get_rank(environ) == 0:
    #     import ipdb; ipdb.set_trace()
    # torch.distributed.barrier()

    seq_data = cast(SequenceData, collater(to_collate))

    batch = SequenceBatch(
        seq_data["seqs"]["seqs"],
        seq_data["seqs"]["seq_lens"],
        target_mask=seq_data["target_loss_mask"]["seqs"],
    )
    batch.to(device)

    return batch


def copy_state(src_module: nn.Module, tgt_module: nn.Module):
    tgt_state = tgt_module.state_dict()  # assumed tgt is not sharded
    for name, src_param in src_module.named_parameters():
        name_edited = name.replace("_checkpoint_wrapped_module.", "")
        if name_edited not in tgt_state.keys():
            raise NameError(f"{name_edited} doesnt exist in tgt_module")
        tgt_param = tgt_state[name_edited]
        tgt_param.data.copy_(src_param.data.to(tgt_param.device))


def find_first_value(lst, value):
    return next((i for i, x in enumerate(lst) if x == value), None)


def generate_rollouts(
    prompts: List[List[int]],
    dp_gang,
    vllm_model: RemoteVllmModel,
    sampling_params=None,
):
    prompts_to_generate = [None] * dp_gang.size
    if dp_gang.rank == 0:
        dp_gang.gather_object(prompts, prompts_to_generate, 0)
    else:
        dp_gang.gather_object(prompts, None, 0)
    if dp_gang.rank == 0:
        rank_batch_sizes = [len(l) for l in prompts_to_generate]
        flat_request_list = []
        for rank_prompts in prompts_to_generate:
            flat_request_list.extend(rank_prompts)

        rollouts = vllm_model.rollout_from_model(
            flat_request_list, sampling_params=sampling_params
        )

        rollouts_to_scatter = []
        rollouts_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rollouts_to_scatter.append(rollouts[rank_start:rank_end])
        dp_gang.scatter_object_list(
            rollouts_per_rank, rollouts_to_scatter, source_rank=0
        )
    else:
        rollouts_per_rank = [None]
        dp_gang.scatter_object_list(rollouts_per_rank, None, source_rank=0)
    dp_gang.barrier()

    return rollouts_per_rank[0]


def generate_rewards(
    prompts: List[List[int]],
    dp_gang,
    vllm_model,
    sampling_params=None,
):
    prompts_to_generate = [None] * dp_gang.size
    if dp_gang.rank == 0:
        dp_gang.gather_object(prompts, prompts_to_generate, 0)
    else:
        dp_gang.gather_object(prompts, None, 0)
    if dp_gang.rank == 0:
        rank_batch_sizes = [len(l) for l in prompts_to_generate]
        flat_request_list = []
        for rank_prompts in prompts_to_generate:
            flat_request_list.extend(rank_prompts)

        rewards = vllm_model.reward_from_model(flat_request_list)

        rewards_to_scatter = []
        rewards_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rewards_to_scatter.append(rewards[rank_start:rank_end])
        dp_gang.scatter_object_list(rewards_per_rank, rewards_to_scatter, source_rank=0)
    else:
        rewards_per_rank = [None]
        dp_gang.scatter_object_list(rewards_per_rank, None, source_rank=0)
    dp_gang.barrier()

    return rewards_per_rank[0]


def generate_rewards_generative(
    prompts: List[str],
    dp_gang,
    vllm_model,
):
    prompts_to_generate = [None] * dp_gang.size
    if dp_gang.rank == 0:
        dp_gang.gather_object(prompts, prompts_to_generate, 0)
    else:
        dp_gang.gather_object(prompts, None, 0)
    if dp_gang.rank == 0:
        rank_batch_sizes = [len(l) for l in prompts_to_generate]
        flat_request_list = []
        for rank_prompts in prompts_to_generate:
            flat_request_list.extend(rank_prompts)

        rewards = vllm_model.rollout_from_model(flat_request_list, string_input=True)

        rewards_to_scatter = []
        rewards_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rewards_to_scatter.append(rewards[rank_start:rank_end])
        dp_gang.scatter_object_list(rewards_per_rank, rewards_to_scatter, source_rank=0)
    else:
        rewards_per_rank = [None]
        dp_gang.scatter_object_list(rewards_per_rank, None, source_rank=0)
    dp_gang.barrier()

    return rewards_per_rank[0]


def prepare_preference_batch_random_pair(
    prompt_batch: PromptBatch, reward_output: dict, gangs
) -> PreferenceBatch:
    """
    Single & random preference pair from rollouts and rewards
    """

    # reward_output = self.process_rollouts(rollouts, prompt_batch.meta_info[self.answer_key])

    chosen_batch = []
    rejected_batch = []
    prompt_lens = []
    dummy_batch_ids = []  # keep posiitons of dummy pairs here

    # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
    for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
        zip(reward_output["rewards"], reward_output["tokens"])
    ):
        chosen_rollout_position = find_first_value(i_batch_rewards, 1)
        rejected_rollout_position = find_first_value(i_batch_rewards, 0)
        if chosen_rollout_position is None or rejected_rollout_position is None:
            # cant form preference pair when we dont have such rollouts
            # this will be dummy batch and we zero out loss
            chosen_rollout_position = 0
            rejected_rollout_position = 1
            dummy_batch_ids.append(i_batch)
        chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
        rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
        prompt_tokens = prompt_batch.prompts[i_batch]

        chosen_tokens = prompt_tokens + chosen_rollout_tokens
        chosen_batch.append(chosen_tokens)

        rejected_tokens = prompt_tokens + rejected_rollout_tokens
        rejected_batch.append(rejected_tokens)

        prompt_lens.append(len(prompt_tokens))

    filter_batch = lambda batch: [
        item for index, item in enumerate(batch) if index not in dummy_batch_ids
    ]

    if len(dummy_batch_ids) == len(reward_output["tokens"]):
        # entire batch does not have a valid preference pair
        # we use it as dummy batch and zero the loss in the end
        is_bad_batch = True
    else:
        # removing dummy pairs from the batch
        chosen_batch = filter_batch(chosen_batch)
        rejected_batch = filter_batch(rejected_batch)
        prompt_lens = filter_batch(prompt_lens)
        is_bad_batch = False

    prompt_lens = torch.tensor(prompt_lens)

    chosen_batch = [
        torch.tensor(sequence, device=gangs.dp.device) for sequence in chosen_batch
    ]
    chosen_batch = collate_with_target_mask(
        chosen_batch, prompt_lens, device=gangs.dp.device
    )

    rejected_batch = [
        torch.tensor(sequence, device=gangs.dp.device) for sequence in rejected_batch
    ]
    rejected_batch = collate_with_target_mask(
        rejected_batch, prompt_lens, device=gangs.dp.device
    )

    batch = PreferenceBatch(
        chosen=chosen_batch,
        rejected=rejected_batch,
        reference_score_chosen=None,
        reference_score_rejected=None,
    )

    return batch, is_bad_batch


def prepare_group_dpo_batch(
    prompt_batch: PromptBatch, reward_output: dict, gangs
) -> PreferenceBatch:
    """
    In group DPO we want to forward all rollouts, and then match all correct vs incorrect options in the loss
    """

    batch = []
    prompt_lens = []
    rewards = []
    dummy_batch_ids = []  # keep posiitons of dummy pairs here

    batch_size = prompt_batch.batch_size

    # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
    for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
        zip(reward_output["rewards"], reward_output["tokens"])
    ):
        if len(set(i_batch_rewards)) == 1:
            # same reward for all rollouts, we wont be able to use it for pairs
            dummy_batch_ids.append(i_batch)

        for rollout_tokens in i_batch_tokens:
            prompt_rollout_tokens = prompt_batch.prompts[i_batch] + list(rollout_tokens)
            batch.append(prompt_rollout_tokens)
            prompt_lens.append(len(prompt_batch.prompts[i_batch]))
        rewards.extend(i_batch_rewards)

    prompt_lens = torch.tensor(prompt_lens)

    batch = [torch.tensor(sequence, device=gangs.dp.device) for sequence in batch]
    batch = collate_with_target_mask(
        batch, prompt_lens, device=gangs.dp.device
    )  # [batch_size * n_rollout]
    rewards = torch.tensor(rewards, device=gangs.dp.device).view(
        batch_size, -1
    )  # [batch_size * n_rollout]

    return batch, rewards, dummy_batch_ids


def combine_prompts_responses_for_scoring(
    prompt_batch, rollouts: List[RequestOutput], gangs
):
    prompts: List[List[int]]
    prompts = prompt_batch.prompts

    responses = []
    for prompt, req_output in zip(prompts, rollouts):
        rollout_outputs = []
        for output in req_output.outputs:
            prompt_response_tokens = prompt + list(output.token_ids)
            rollout_outputs.append(prompt_response_tokens)
        responses.extend(rollout_outputs)

    # if gangs.root.rank == 0:
    #     from pudb.remote import set_trace
    #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

    # gangs.root.barrier()

    return responses


def convert_vllm_output_to_ref_score(vllm_outputs: List[RequestOutput], gangs):
    ref_scores = []
    for req_output in vllm_outputs:
        prompt_logprobs = req_output.prompt_logprobs[1:]
        logprobs = [list(d.values())[0].logprob for d in prompt_logprobs]
        # selecting only the response part that we scored
        logprobs = torch.tensor(logprobs)
        ref_scores.append(logprobs)

    return ref_scores


def compute_token_level_entropy(logits: torch.Tensor, target_mask: torch.Tensor):
    """Calculate entropy from logits. Returns sum of entropies averages for each sequence."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    entropy_target_only = entropy * target_mask
    entropy_per_seq = entropy_target_only.sum(dim=-1) / target_mask.sum(dim=-1)

    return entropy_per_seq


def log_rollouts(prompt_batch: PromptBatch, rollouts, split_name, num_rollouts=1):
    """
    log the first num_rollouts rollouts for first prompt in the batch
    """
    if "prompt_raw" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("prompt_raw")[0]
    elif "raw_prompt" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("raw_prompt")[0]
    else:
        # raw text prompt doesn't exist for this dataset
        prompt = "DUMMY PROMPT"

    log.info(f"{split_name} Prompt: {prompt}")
    for rollout in rollouts[0].outputs[:num_rollouts]:
        rollout_text = rollout.text
        log.info(f"{split_name} Rollout: {rollout_text}")


def log_rollout_tree(
    srt: ScoredRolloutTree,
    prefix: str,
    seed_idx: int = 0,
    max_branching_factor: int = 2,
):
    b = seed_idx
    log.info(f"[{prefix}] Logging rollout tree for seed index {b}")
    log.info(f"[{prefix}] Seed (s): {srt.s_text_B[b]}")
    log.info(f"[{prefix}] Embedded Seed (s_emb): {srt.s_emb_text_B[b]}")

    for a in range(min(srt.A, max_branching_factor)):
        log.info(
            f"[{prefix}] (reward = {srt.x_reward_BA[b][a]}, reward patched: {srt.x_reward_patched_BA[b][a]}, error = {srt.x_error_BA[b][a]}, advantage = {srt.x_advantage_BA[b][a]})] Raw prompt (x_raw_{a}): {srt.x_raw_text_BA[b][a]}"
        )
        log.info(
            f"[{prefix}] (reward = {srt.x_reward_BA[b][a]}, reward patched: {srt.x_reward_patched_BA[b][a]}, error = {srt.x_error_BA[b][a]}, advantage = {srt.x_advantage_BA[b][a]})] Embedded prompt (x_emb_{a}): {srt.x_emb_text_BA[b][a]}"
        )

        for d in range(min(srt.D, max_branching_factor)):
            log.info(
                f"[{prefix}] (reward = {srt.y_reward_BAD[b][a][d]}, reward patched: {srt.y_reward_patched_BAD[b][a][d]}, error = {srt.y_error_BAD[b][a][d]}, advantage = {srt.y_advantage_BAD[b][a][d]})] Raw response (y_{a}^{d}): {srt.y_raw_text_BAD[b][a][d]}"
            )


def log_rollout_tree_direct(
    srtd: ScoredRolloutTreeDirect,
    prefix: str,
    seed_idx: int = 0,
    max_branching_factor: int = 2,
):
    b = seed_idx
    log.info(f"[{prefix}] Logging direct rollout tree for seed index {b}")
    log.info(f"[{prefix}] Seed (s): {srtd.s_text_B[b]}")
    log.info(f"[{prefix}] Embedded Seed (s_emb): {srtd.s_emb_text_B[b]}")

    for d in range(min(srtd.D, max_branching_factor)):
        log.info(
            f"[{prefix}] (reward = {srtd.y_reward_BD[b][d]}, reward patched: {srtd.y_reward_patched_BD[b][d]}, error = {srtd.y_error_BD[b][d]}, advantage = {srtd.y_advantage_BD[b][d]})] Raw response (y_direct^{d}): {srtd.y_raw_text_BD[b][d]}"
        )


def get_rollout_lengths(rollouts: List[SequenceData]):
    """Get the lengths of the rollouts."""
    rollout_lengths = []
    for rollout in rollouts:
        for sample in rollout.outputs:
            token_ids = sample.token_ids
            token_ids_len = len(token_ids)
            rollout_lengths.append(token_ids_len)
    return rollout_lengths


def update_avg_length_filtered(
    metric_bag: MetricBag, tokens_flat: List[List[int]], name: str
):
    rollout_lengths = [len(tokens) for tokens in tokens_flat if tokens is not None]
    if len(rollout_lengths) > 0:
        avg_rollout_length = torch.tensor(rollout_lengths).float().mean()
        metric_bag.get(Mean, name).update(
            avg_rollout_length, weight=len(rollout_lengths)
        )
    else:
        metric_bag.get(Mean, name).update(torch.tensor(0.0), weight=0)


@torch.inference_mode()
def update_avg_value_filtered(
    metric_bag: MetricBag, values_flat: List[bool] | List[float], name: str
):
    values_filtered = [v for v in values_flat if v is not None]
    if len(values_filtered) > 0:
        avg_value = torch.tensor(values_filtered).float().mean()
        metric_bag.get(Mean, name).update(avg_value, weight=len(values_filtered))
    else:
        metric_bag.get(Mean, name).update(torch.tensor(0.0), weight=0)


@torch.inference_mode()
def update_std_value_filtered(
    metric_bag: MetricBag, values_flat: List[bool] | List[float], name: str
):
    values_filtered = [v for v in values_flat if v is not None]
    if len(values_filtered) > 0:
        std_value = torch.tensor(values_filtered).float().std()
        metric_bag.get(Mean, name).update(std_value, weight=len(values_filtered))
    else:
        metric_bag.get(Mean, name).update(torch.tensor(0.0), weight=0)


@torch.inference_mode()
def update_logit_entropy(
    metric_bag: MetricBag, logit_entropy: Tensor, prefix: str = ""
) -> None:
    # logit_entropy is expected to contain token-level entropy for every sequence in the current batch
    batch_size = logit_entropy.size(0)
    metric_bag.get(Mean, prefix + "logit_entropy").update(
        logit_entropy.sum() / batch_size, weight=batch_size
    )


@torch.inference_mode()
def update_dpo_loss(
    metric_bag: MetricBag,
    loss: Tensor,
    batch_size: int,
    weight_factor: float = 1.0,
    prefix: str = "",
) -> None:
    metric_bag.get(Mean, prefix + "dpo_loss").update(
        loss / batch_size, weight=batch_size * weight_factor
    )


@torch.inference_mode()
def update_avg_dpo_grad_factor(
    metric_bag: MetricBag,
    grad_factor: Tensor,
    batch_size: int,
    weight_factor: float = 1.0,
    prefix: str = "",
) -> None:
    metric_bag.get(Mean, prefix + "avg_dpo_grad_factor").update(
        grad_factor, weight=batch_size * weight_factor
    )


@torch.inference_mode()
def update_num_dummy_batches(
    metric_bag: MetricBag, batch: PreferenceBatch, num_dummy_batches: int
):
    metric_bag.get(Mean, "num_dummy_batches").update(
        num_dummy_batches / batch.chosen.batch_size, weight=batch.chosen.batch_size
    )


@torch.inference_mode()
def update_avg_reward(metric_bag: MetricBag, avg_reward, prefix: str = ""):
    metric_bag.get(Mean, prefix + "avg_reward").update(avg_reward, weight=1)


@torch.inference_mode()
def update_std_reward(metric_bag: MetricBag, std_reward, prefix: str = ""):
    metric_bag.get(Mean, prefix + "std_reward").update(std_reward, weight=1)


@torch.inference_mode()
def update_avg_faithful(metric_bag: MetricBag, avg_faithful, prefix: str = ""):
    metric_bag.get(Mean, prefix + "avg_faithful").update(avg_faithful, weight=1)


@torch.inference_mode()
def update_avg_parsing_error(
    metric_bag: MetricBag, avg_parsing_error, weight: int = 1, prefix: str = ""
):
    metric_bag.get(Mean, prefix + "avg_parsing_error").update(
        avg_parsing_error, weight=weight
    )


@torch.inference_mode()
def update_avg_max_length_error(
    metric_bag: MetricBag, avg_max_length_error, weight: int = 1, prefix: str = ""
):
    metric_bag.get(Mean, prefix + "avg_max_length_error").update(
        avg_max_length_error, weight=weight
    )


@torch.inference_mode()
def update_avg_rollout_length(
    metric_bag: MetricBag, avg_rollout_length, prefix: str = ""
):
    metric_bag.get(Mean, prefix + "avg_rollout_length").update(
        avg_rollout_length, weight=1
    )


@torch.inference_mode()
def update_avg_length(
    metric_bag: MetricBag, avg_length, weight: int = 1, name: str = ""
):
    metric_bag.get(Mean, name + "_avg_length").update(avg_length, weight=weight)


@torch.inference_mode()
def update_avg_reward_len_norm(
    metric_bag: MetricBag, avg_reward_len_norm, prefix: str = ""
):
    metric_bag.get(Mean, prefix + "avg_reward_len_norm").update(
        avg_reward_len_norm, weight=1
    )


@torch.inference_mode()
def update_avg_loss_zeroer(metric_bag: MetricBag, avg_loss_zeroer, prefix: str = ""):
    metric_bag.get(Mean, prefix + "avg_loss_zeroer").update(avg_loss_zeroer, weight=1)


@torch.inference_mode()
def update_batch_metrics(
    metric_bag: MetricBag,
    batch: PreferenceBatch,
    train: bool,
    prefix: str = "",
) -> None:
    num_examples = batch.batch_size
    metric_bag.get(Sum, prefix + "num_examples").update(num_examples)
    if train:
        metric_bag.get(Sum, prefix + "total_num_examples").update(num_examples)


def update_grpo_batch_metrics(
    metric_bag: MetricBag, batch: SequenceBatch, train=True, prefix: str = ""
) -> None:
    metric_bag.get(Sum, prefix + "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, prefix + "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, prefix + "num_target_elements").update(
        batch.num_target_elements
    )

    metric_bag.get(Sum, prefix + "padding").update(batch.padding)
    if train:
        metric_bag.get(Sum, prefix + "total_num_examples").update(batch.num_examples)

        metric_bag.get(Sum, prefix + "total_num_elements").update(batch.num_elements)

        metric_bag.get(Sum, prefix + "total_num_target_elements").update(
            batch.num_target_elements
        )


@torch.inference_mode()
def update_grpo_loss(
    metric_bag: MetricBag, batch_size: int, loss: Tensor, prefix: str = ""
) -> None:
    """Update the GRPO loss metric.

    :param batch:
        The batch processed by the model.
    :param loss:
        The GRPO loss of ``batch``.
    """
    metric_bag.get(Mean, prefix + "grpo_loss").update(
        loss / batch_size, weight=batch_size
    )


def compute_reference_logps(
    gangs: Gangs,
    reference_model: RemoteVllmModel,
    seqs: torch.Tensor,
    layout: BatchLayout,
    prompt_lengths: List[int],
):
    seqs_to_score = seqs.tolist()
    if layout.padded:
        padding_mask = layout.position_indices >= 0  # True when non-pad
        seqs_to_score = [
            seq[:l] for seq, l in zip(seqs_to_score, padding_mask.sum(-1).tolist())
        ]

    scored_responses = generate_rollouts(
        seqs_to_score, dp_gang=gangs.dp, vllm_model=reference_model
    )
    ref_logps = convert_vllm_output_to_ref_score(scored_responses, gangs)
    ref_logps = collate_with_target_mask(
        ref_logps, prompt_lengths, device=gangs.dp.device
    ).seqs

    return ref_logps


def get_parameter_converter(model_config):
    from fairseq2.models.llama import LLaMAConfig
    from fairseq2.models.qwen import QwenConfig

    if isinstance(model_config, QwenConfig):
        from fairseq2.models.qwen._hg import _convert_parameter
    elif isinstance(model_config, LLaMAConfig):
        from fairseq2.models.llama._hg import _convert_parameter
    else:
        raise RuntimeError(f"{model_config} not supported in online recipe")

    return _convert_parameter
