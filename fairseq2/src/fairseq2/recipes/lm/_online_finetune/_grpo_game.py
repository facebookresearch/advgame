# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Final, Tuple, final

import torch
from torch import Tensor
from typing_extensions import override
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from fairseq2.context import RuntimeContext
from fairseq2.datasets import SequenceBatch
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.recipes import Model, TrainUnit
from fairseq2.recipes.lm._online_finetune._common import (
    ScoredRolloutTree,
    ScoredRolloutTreeDirect,
    StatefulRolloutBagGame,
    VllmGameSyncSection,
    _collate,
    compute_token_level_entropy,
    rollout_to_vllm_logps,
    update_batch_metrics,
    update_grpo_batch_metrics,
    update_grpo_loss,
    update_logit_entropy,
    xError,
    yError,
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_model import maybe_sync_model
from fairseq2.recipes.lm._online_finetune._rewards import (
    GRPOBatch,
    RewardGameSection,
    VLLMOutputRewardGame,
    VLLMOutputRewardGameHandler,
)
from fairseq2.utils.list import linit
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import ValidationError, ValidationResult, validate


@final
class GrpoGameFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _step_nr: int
    _valid_step_nr: int
    _model_attacker: Model
    _model_defender: Model
    _ema_model_attacker: Model
    _ema_model_defender: Model
    _ref_model_attacker: Model
    _ref_model_defender: Model
    _config: GrpoGameFinetuneConfig
    _model_update_group: PyNcclCommunicator
    _reward: VLLMOutputRewardGame
    _display_name: str
    _rollout_bag: StatefulRolloutBagGame

    def __init__(
        self,
        model_attacker: Model,
        model_defender: Model,
        ema_model_attacker: Model,
        ema_model_defender: Model,
        ref_model_attacker: Model,
        ref_model_defender: Model,
        reward: VLLMOutputRewardGame,
        gangs: Gangs,
        config: GrpoGameFinetuneConfig,
    ) -> None:
        super().__init__()
        self._model_attacker = model_attacker
        self._model_defender = model_defender
        self._ema_model_attacker = ema_model_attacker
        self._ema_model_defender = ema_model_defender
        self._ref_model_attacker = ref_model_attacker
        self._ref_model_defender = ref_model_defender
        self._config = config
        self._gangs = gangs
        self._reward = reward
        self._step_nr = None

        self._ema_model_attacker.module.eval()
        self._ema_model_defender.module.eval()
        self._ref_model_attacker.module.eval()
        self._ref_model_defender.module.eval()

        self._rollout_bag = StatefulRolloutBagGame(
            num_microbatches_B=config.loss_config.num_microbatches_B,
            num_microbatches_A=config.loss_config.num_microbatches_A,
            num_microbatches_D=config.loss_config.num_microbatches_D,
        )

        if self._config.vllm_sync.sync_ema_model_instead_of_train_model:
            self._sync_model_attacker = self._ema_model_attacker
            self._sync_model_defender = self._ema_model_defender
        else:
            self._sync_model_attacker = self._model_attacker
            self._sync_model_defender = self._model_defender

        self._display_name = "GRPO_game"

    @property
    @override
    def display_name(self) -> str | None:
        return self._display_name

    def set_train_step_nr(self, train_step_nr: int) -> None:
        self._valid_step_nr = train_step_nr

    def finalize(self, metric_bag: MetricBag) -> None:
        pass

    @property
    @override
    def name(self) -> str | None:
        return self._display_name

    @override
    def __call__(
        self,
        prompt_batch: PromptBatch,
        metric_bag: MetricBag,
        prefix: str,
        skip_attacker: bool,
        train: bool,
    ) -> Tuple[Tensor, Tensor, int, int]:
        log.info(
            f"[{prefix}] Starting AdvGameGRPO call at step nr: {self._step_nr} with batch size {prompt_batch.batch_size} and skip_attacker={skip_attacker}"
        )

        update_batch_metrics(
            metric_bag,
            prompt_batch,
            train=train,
            prefix=prefix,
        )

        if not train:
            srt: ScoredRolloutTree | ScoredRolloutTreeDirect
            srt = self._reward.generate_scored_rollout_tree(
                s_prompt_batch=prompt_batch,
                train=train,
                metric_bag=metric_bag,
                prefix=prefix,
                skip_attacker=skip_attacker,
            )
            return torch.tensor(0.0), torch.tensor(0.0), 0, 0

        self._rollout_bag.maybe_reset_bag(self._step_nr)

        if self._rollout_bag.srt is None:
            if not skip_attacker:
                maybe_sync_model(
                    self._gangs,
                    self._sync_model_attacker,
                    self._reward.vllm_model_attacker,
                    self._step_nr,
                    self._config.vllm_sync.sync_model_every_n_steps,  # 1 by default
                )
            maybe_sync_model(
                self._gangs,
                self._sync_model_defender,
                self._reward.vllm_model_defender,
                self._step_nr,
                self._config.vllm_sync.sync_model_every_n_steps,  # 1 by default
            )

            srt: ScoredRolloutTree | ScoredRolloutTreeDirect
            srt = self._reward.generate_scored_rollout_tree(
                s_prompt_batch=prompt_batch,
                train=train,
                metric_bag=metric_bag,
                prefix=prefix,
                skip_attacker=skip_attacker,
            )
            self._rollout_bag.save(srt)
            srt = self._rollout_bag.load()
        else:
            srt = self._rollout_bag.load()

        batch_attacker: GRPOBatch | None
        batch_defender: GRPOBatch
        batch_attacker, batch_defender = self._prepare_grpo_game_batch(
            srt, metric_bag, prefix=prefix
        )

        if self._config.loss_config.use_ema_model_as_ref_model:
            ref_model_attacker = self._ema_model_attacker
            ref_model_defender = self._ema_model_defender
        else:
            ref_model_attacker = self._ref_model_attacker
            ref_model_defender = self._ref_model_defender

        if batch_attacker is None:
            loss_attacker = torch.tensor(0.0)
            batch_size_attacker = 0
        else:
            loss_attacker, total_tokens_attacker = self._compute_loss(
                batch=batch_attacker,
                model=self._model_attacker,
                ref_model=ref_model_attacker,
                prefix=prefix + "attacker/",
                metric_bag=metric_bag,
            )
            if self._config.loss_config.loss_token_mean:
                batch_size_attacker = total_tokens_attacker
            else:
                batch_size_attacker = batch_attacker.rewards.size(0)

        loss_defender, total_tokens_defender = self._compute_loss(
            batch=batch_defender,
            model=self._model_defender,
            ref_model=ref_model_defender,
            prefix=prefix + "defender/",
            metric_bag=metric_bag,
        )
        if self._config.loss_config.loss_token_mean:
            batch_size_defender = total_tokens_defender
        else:
            batch_size_defender = batch_defender.rewards.size(0)

        log.info(
            f"[{prefix}] Finished AdvGameGRPO call. Loss attacker: {loss_attacker.item()}, loss defender: {loss_defender.item()}, batch_size_attacker: {batch_size_attacker}, batch_size_defender: {batch_size_defender}"
        )
        return loss_attacker, loss_defender, batch_size_attacker, batch_size_defender

    def _compute_loss(
        self,
        batch: GRPOBatch,
        model: Model,
        ref_model: Model,
        prefix: str,
        metric_bag: MetricBag,
    ) -> Tensor:
        update_grpo_batch_metrics(
            metric_bag, batch.prompt_rollouts, train=True, prefix=prefix
        )

        (
            grpo_input_batch,
            grpo_target_batch,
        ) = batch.prompt_rollouts.as_auto_regressive()
        (
            grpo_input_batch_seqs,
            grpo_input_batch_seqs_layout,
        ) = grpo_input_batch.as_input()

        grpo_model_logits = model.module(
            grpo_input_batch_seqs, grpo_input_batch_seqs_layout
        )
        model_logps = self._gather_lprobs(grpo_model_logits, grpo_target_batch)
        with torch.no_grad():
            ref_logits = ref_model.module(
                grpo_input_batch_seqs, grpo_input_batch_seqs_layout
            )
            ref_logps = self._gather_lprobs(ref_logits, grpo_target_batch)

        tgt_logit_entropy = compute_token_level_entropy(
            grpo_model_logits, grpo_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum()
            * self._config.loss_config.entropy_regularizer_scale
        )
        update_logit_entropy(metric_bag, tgt_logit_entropy, prefix=prefix)

        _grpo_objective, total_tokens = self._compute_grpo_objective(
            model_logps=model_logps,
            vllm_logps=batch.vllm_logps,
            ref_logps=ref_logps,
            advantages=batch.rewards,
            dummy_mask=batch.dummy_mask,
            target_batch=grpo_target_batch,
        )

        grpo_loss = -_grpo_objective + max_entropy_regularizer

        batch_size = len(batch.rewards)
        update_grpo_loss(metric_bag, batch_size, grpo_loss, prefix=prefix)

        return grpo_loss, total_tokens

    def _gather_lprobs(self, logits: Tensor, target: SequenceBatch) -> Tensor:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )  # [Batch, 1]

        return per_token_logps

    def _compute_grpo_objective(
        self,
        model_logps: Tensor,
        vllm_logps: Tensor,
        ref_logps: Tensor,
        advantages: Tensor,  # outcome based only for now
        dummy_mask: Tensor,  # 0 if dummy, 1 if not
        target_batch: SequenceBatch,
    ) -> Tensor:
        batch_size = advantages.size(0)
        num_rollouts = advantages.size(1)

        model_logps = model_logps.view(batch_size, num_rollouts, -1)
        vllm_logps = vllm_logps.view(batch_size, num_rollouts, -1).to(
            model_logps.device
        )
        target_mask = target_batch.target_mask.view(batch_size, num_rollouts, -1)

        assert dummy_mask.shape == (batch_size, num_rollouts)
        assert advantages.shape == (batch_size, num_rollouts)

        per_token_scaled_advantage = (
            model_logps - model_logps.detach()
        ).exp() * advantages[:, :, None]

        if self._config.loss_config.use_importance_sampling_correction:
            tis_imp_ratio = torch.exp(model_logps - vllm_logps)
            tis_imp_ratio = torch.clamp(
                tis_imp_ratio, max=self._config.loss_config.tis_imp_ratio_cap
            )
            per_token_scaled_advantage = per_token_scaled_advantage * tis_imp_ratio

        if ref_logps is not None:
            ref_logps = ref_logps.view(batch_size, num_rollouts, -1)

            # kl penalty
            kl = (ref_logps - model_logps).exp() - (ref_logps - model_logps) - 1.0
            per_token_loss = (
                per_token_scaled_advantage - self._config.loss_config.beta * kl
            )
        else:
            per_token_loss = per_token_scaled_advantage

        total_tokens = (target_mask * dummy_mask[:, :, None]).sum().item()

        if self._config.loss_config.length_normalization:
            per_rollout_loss = (per_token_loss * target_mask).sum(
                dim=-1
            ) / target_mask.sum(dim=-1)
            per_seq_loss = (per_rollout_loss * dummy_mask).mean(dim=1)
        elif self._config.loss_config.loss_token_mean:
            per_seq_loss = per_token_loss * target_mask * dummy_mask[:, :, None]
        else:
            per_rollout_loss = (per_token_loss * target_mask).sum(dim=-1)
            per_seq_loss = (per_rollout_loss * dummy_mask).mean(dim=1)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        return per_seq_loss.sum(), total_tokens

    def _prepare_grpo_game_batch(
        self,
        srt: ScoredRolloutTree | ScoredRolloutTreeDirect,
        metric_bag: MetricBag,
        prefix: str = "",
    ) -> Tuple[GRPOBatch, GRPOBatch]:
        if isinstance(srt, ScoredRolloutTreeDirect):
            return None, self._prepare_grpo_game_batch_direct(
                srt, metric_bag, prefix=prefix
            )

        log.info(f"[{prefix}] Preparing GRPO batches from scored rollout tree")
        # reward_output = self.process_rollouts(rollout_tree, metric_bag, prefix=prefix)

        B, A, D = (srt.B, srt.A, srt.D)
        K = B * A
        # L = K * D

        # TODO: more elegant way to do this?
        dummy_tokens = [0] * 10
        dummy_vllm_logps = torch.zeros(9)
        dummy_length = 5
        dummy_advantage = 0.0

        sx_tokens_K = []
        sx_vllm_logps_K = []
        s_emb_lens_K = []
        x_advantage_BA = linit([B, A])
        dummy_mask_BA = linit([B, A])

        xy_tokens_L = []
        xy_vllm_logps_L = []
        x_emb_lens_L = []
        y_advantage_KD = linit([K, D])
        dummy_mask_KD = linit([K, D])

        for b, a in product(range(B), range(A)):
            if srt.x_error_BA[b][a] == xError.NO_ERROR:
                sx_tokens_K.append(srt.s_emb_tokens_B[b] + srt.x_raw_tokens_BA[b][a])
                sx_vllm_logps_K.append(rollout_to_vllm_logps(srt.x_rollout_BA[b][a]))
                x_advantage_BA[b][a] = srt.x_advantage_BA[b][a]
                s_emb_lens_K.append(len(srt.s_emb_tokens_B[b]))
                dummy_mask_BA[b][a] = 1.0
            else:
                sx_tokens_K.append(dummy_tokens)
                sx_vllm_logps_K.append(dummy_vllm_logps)
                s_emb_lens_K.append(dummy_length)
                x_advantage_BA[b][a] = dummy_advantage
                dummy_mask_BA[b][a] = 0.0

            for d in range(D):
                if srt.y_error_BAD[b][a][d] == yError.NO_ERROR:
                    xy_tokens_L.append(
                        srt.x_emb_tokens_BA[b][a] + srt.y_raw_tokens_BAD[b][a][d]
                    )
                    xy_vllm_logps_L.append(
                        rollout_to_vllm_logps(srt.y_rollout_BAD[b][a][d])
                    )
                    y_advantage_KD[b * A + a][d] = srt.y_advantage_BAD[b][a][d]
                    x_emb_lens_L.append(len(srt.x_emb_tokens_BA[b][a]))
                    dummy_mask_KD[b * A + a][d] = 1.0
                else:
                    xy_tokens_L.append(dummy_tokens)
                    xy_vllm_logps_L.append(dummy_vllm_logps)
                    x_emb_lens_L.append(dummy_length)
                    y_advantage_KD[b * A + a][d] = dummy_advantage
                    dummy_mask_KD[b * A + a][d] = 0.0

        sx_vllm_logps_max_len = max(t.size(0) for t in sx_vllm_logps_K)
        sx_vllm_logps_tensor_BA = torch.zeros(
            len(sx_vllm_logps_K), sx_vllm_logps_max_len
        )
        for i, t in enumerate(sx_vllm_logps_K):
            sx_vllm_logps_tensor_BA[i, : t.size(0)] = t

        xy_vllm_logps_max_len = max(t.size(0) for t in xy_vllm_logps_L)
        xy_vllm_logps_tensor_KD = torch.zeros(
            len(xy_vllm_logps_L), xy_vllm_logps_max_len
        )
        for i, t in enumerate(xy_vllm_logps_L):
            xy_vllm_logps_tensor_KD[i, : t.size(0)] = t

        x_advantage_tensor_BA = torch.tensor(
            x_advantage_BA, device=self._gangs.dp.device
        ).float()

        dummy_mask_tensor_BA = torch.tensor(
            dummy_mask_BA, device=self._gangs.dp.device
        ).float()

        y_advantage_tensor_KD = torch.tensor(
            y_advantage_KD, device=self._gangs.dp.device
        ).float()

        dummy_mask_tensor_KD = torch.tensor(
            dummy_mask_KD, device=self._gangs.dp.device
        ).float()

        sx_collated_K = _collate(
            sx_tokens_K, s_emb_lens_K, device=self._gangs.dp.device
        )
        xy_collated_L = _collate(
            xy_tokens_L, x_emb_lens_L, device=self._gangs.dp.device
        )

        grpo_batch_attacker = GRPOBatch(
            prompt_rollouts=sx_collated_K,
            rewards=x_advantage_tensor_BA,
            prompt_lengths=s_emb_lens_K,
            dummy_mask=dummy_mask_tensor_BA,
            vllm_logps=sx_vllm_logps_tensor_BA,
        )
        grpo_batch_defender = GRPOBatch(
            prompt_rollouts=xy_collated_L,
            rewards=y_advantage_tensor_KD,
            prompt_lengths=x_emb_lens_L,
            dummy_mask=dummy_mask_tensor_KD,
            vllm_logps=xy_vllm_logps_tensor_KD,
        )
        log.info(f"[{prefix}] Finished preparing GRPO batches.")
        return grpo_batch_attacker, grpo_batch_defender

    def _prepare_grpo_game_batch_direct(
        self,
        srtd: ScoredRolloutTreeDirect,
        metric_bag: MetricBag,
        prefix: str = "",
    ) -> GRPOBatch:
        log.info(f"[{prefix}] Preparing direct GRPO batch from scored rollout tree")
        B, D = srtd.B, srtd.D
        # F = B * D  # total number of rollouts for defender

        # TODO: more elegant way to do this?
        dummy_tokens = [0] * 10
        dummy_vllm_logps = torch.zeros(9)
        dummy_length = 5
        dummy_advantage = 0.0

        sy_tokens_F = []
        sy_vllm_logps_F = []
        s_emb_lens_F = []
        y_advantage_BD = linit([B, D])
        dummy_mask_BD = linit([B, D])
        for b, d in product(range(B), range(D)):
            if srtd.y_error_BD[b][d] == yError.NO_ERROR:
                sy_tokens_F.append(srtd.s_emb_tokens_B[b] + srtd.y_raw_tokens_BD[b][d])
                sy_vllm_logps_F.append(rollout_to_vllm_logps(srtd.y_rollout_BD[b][d]))
                s_emb_lens_F.append(len(srtd.s_emb_tokens_B[b]))
                y_advantage_BD[b][d] = srtd.y_advantage_BD[b][d]
                dummy_mask_BD[b][d] = 1.0
            else:
                sy_tokens_F.append(dummy_tokens)
                sy_vllm_logps_F.append(dummy_vllm_logps)
                s_emb_lens_F.append(dummy_length)
                y_advantage_BD[b][d] = dummy_advantage
                dummy_mask_BD[b][d] = 0.0

        s_vllm_logps_max_len = max(t.size(0) for t in sy_vllm_logps_F)
        sy_vllm_logps_tensor_F = torch.zeros(len(sy_vllm_logps_F), s_vllm_logps_max_len)
        for i, t in enumerate(sy_vllm_logps_F):
            sy_vllm_logps_tensor_F[i, : t.size(0)] = t

        y_advantage_tensor_BD = torch.tensor(
            y_advantage_BD, device=self._gangs.dp.device
        ).float()

        dummy_mask_tensor_BD = torch.tensor(
            dummy_mask_BD, device=self._gangs.dp.device
        ).float()

        sy_collated_F = _collate(
            sy_tokens_F, s_emb_lens_F, device=self._gangs.dp.device
        )

        grpo_batch_attacker = GRPOBatch(
            prompt_rollouts=sy_collated_F,
            rewards=y_advantage_tensor_BD,
            prompt_lengths=s_emb_lens_F,
            dummy_mask=dummy_mask_tensor_BD,
            vllm_logps=sy_vllm_logps_tensor_F,
        )
        log.info(f"[{prefix}] Finished preparing direct GRPO batch.")
        return grpo_batch_attacker

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def model_attacker(self) -> Model:
        return self._model_attacker

    @property
    @override
    def model_defender(self) -> Model:
        return self._model_defender

    @property
    @override
    def ema_model_attacker(self) -> Model | None:
        return self._ema_model_attacker

    @property
    @override
    def ema_model_defender(self) -> Model | None:
        return self._ema_model_defender

    @property
    @override
    def ref_model_attacker(self) -> Model | None:
        return self._ref_model_attacker

    @property
    @override
    def ref_model_defender(self) -> Model | None:
        return self._ref_model_defender


GRPO_GAME_FINETUNE_UNIT: Final = "grpo_game"


@dataclass(kw_only=True)
class GrpoGameLossConfig:
    # group_size_attacker: int = 4
    # group_size_defender: int = 4
    # """Number of responses to sample per prompt for advantage computation.

    # This value must match the 'n' parameter in the VLLM sampling params.
    # """

    # forward_group_size_attacker: int = 2
    # forward_group_size_defender: int = 2
    # """Maximum number of responses to process in a single forward pass.

    # When group_size > forward_group_size, responses are processed in multiple micro-batches
    # to reduce memory usage (similar to gradient accumulation). Each micro-batch processes
    # forward_group_size responses and accumulates gradients until all group_size responses
    # are processed. `forward_group_size_attacker` should evenly divide `rollout_tree.num_attacker_samples`.
    # `forward_group_size_defender` should evenly divide `rollout_tree.num_defender_samples`.
    # """

    num_microbatches_B: int = 1
    num_microbatches_A: int = 1
    num_microbatches_D: int = 1
    """Number of mini-batches used over dimensions B, A, D for the rollout bag."""

    beta: float = 0.001
    """The coefficient of regularization towards the reference model."""

    entropy_regularizer_scale: float = 0.0
    """Scale factor for entropy regularization term."""

    length_normalization: bool = True
    """If True, normalize loss by sequence length. If False, use sequence-level loss."""

    use_ema_model_as_ref_model: bool = False
    """Use the EMA model as the reference model instead of the provided reference model."""

    loss_token_mean: bool = False
    """If True, average loss over tokens. If False, sum over tokens."""

    use_importance_sampling_correction: bool = False
    """If True, apply importance sampling correction using the VLLM model's logprobs"""

    tis_imp_ratio_cap: float = 2.0
    """Maximum cap for the truncated importance sampling ratio."""

    def validate(self) -> None:
        result = ValidationResult()

        if self.num_microbatches_B <= 0:
            result.add_error("`num_microbatches_B` must be > 0.")

        if self.num_microbatches_A <= 0:
            result.add_error("`num_microbatches_A` must be > 0.")

        if self.num_microbatches_D <= 0:
            result.add_error("`num_microbatches_D` must be > 0.")

        if result.has_error:
            raise ValidationError(
                "The loss configuration section has one or more validation errors:",
                result,  # fmt: skip
            )


@dataclass(kw_only=True)
class GrpoGameFinetuneConfig:
    """Configuration for Generalized Reward-Paired Optimization (GRPO) finetuning.

    GRPO finetuning uses a policy model to generate diverse responses, which are then
    evaluated by a reward model. The policy is trained to maximize the expected reward
    while maintaining proximity to a reference model.
    """

    loss_config: GrpoGameLossConfig = field(
        default_factory=lambda: GrpoGameLossConfig()
    )
    """Configuration for GRPO loss computation, including rollout handling and regularization."""

    reward: RewardGameSection = field(
        default_factory=lambda: RewardGameSection(name="generative_game_verifier")
    )
    """Configuration for the reward function that evaluates generated rollouts."""

    vllm_sync: VllmGameSyncSection = field(
        default_factory=lambda: VllmGameSyncSection()
    )

    def validate(self) -> None:
        result = ValidationResult()

        if (
            self.reward.config.rollout_tree.num_attacker_samples
            % self.loss_config.num_microbatches_A
            != 0
        ):
            result.add_error(
                "`rollout_tree.num_attacker_samples` must be divisible by `loss_config.num_microbatches_A`."
            )

        if (
            self.reward.config.rollout_tree.num_defender_samples
            % self.loss_config.num_microbatches_D
            != 0
        ):
            result.add_error(
                "`rollout_tree.num_defender_samples` must be divisible by `loss_config.num_microbatches_D`."
            )

        if (
            self.loss_config.use_importance_sampling_correction
            and self.vllm_sync.sync_ema_model_instead_of_train_model
        ):
            result.add_error(
                "Should only use importance sampling correction when doing on-policy generation (i.e., `vllm_sync.sync_ema_model_instead_of_train_model` is False)."
            )

        if result.has_error:
            raise ValidationError(
                "The GRPO finetuning configuration section has one or more validation errors:",
                result,  # fmt: skip
            )


@final
class GrpoGameFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    """
    Handles creation and configuration of GRPO fine-tuning units.
    """

    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self,
        model_attacker: Model,
        model_defender: Model,
        ema_model_attacker: Model,
        ema_model_defender: Model,
        ref_model_attacker: Model,
        ref_model_defender: Model,
        gangs: Gangs,
        recipe_config: object,
        vllm_actors: object,
    ) -> TrainUnit[PreferenceBatch]:
        config = structure(recipe_config.criterion.config, GrpoGameFinetuneConfig)

        validate(config)
        log.info(f"GRPO Game loss config:\n{config}")

        reward_registry = self._context.get_registry(VLLMOutputRewardGameHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            vllm_actors=vllm_actors,
            reward_config=config.reward.config,
            gangs=gangs,
            context=self._context,
        )

        # TODO: add the force syncing back here, make sync mode passable or use force_sync?
        # # sync models here before we start training
        # if config.vllm_sync.sync_model_every_n_steps > 0:
        #     maybe_sync_model(gangs, model, vllm_model, -1, -1, force_sync=True)
        # if config.vllm_sync.sync_ref_model_every_n_steps > 0:
        #     maybe_sync_model(gangs, model, reference_model, -1, -1, force_sync=True)

        # TODO: decide converter as part of the model handler
        # TODO: this was missing in base GRPO, double check
        for model in [
            model_attacker,
            model_defender,
            ema_model_attacker,
            ema_model_defender,
        ]:
            if "llama" in model.name:
                from fairseq2.models.llama._hg import _convert_parameter

                model._convert_parameter = _convert_parameter
            elif "qwen" in model.name:
                from fairseq2.models.qwen._hg import _convert_parameter

                model._convert_parameter = _convert_parameter
            else:
                raise RuntimeError

        log.info("GRPO Game setup complete.")

        return GrpoGameFinetuneUnit(
            model_attacker=model_attacker,
            model_defender=model_defender,
            ema_model_attacker=ema_model_attacker,
            ema_model_defender=ema_model_defender,
            ref_model_attacker=ref_model_attacker,
            ref_model_defender=ref_model_defender,
            reward=reward,
            gangs=gangs,
            config=config,
        )

    @property
    @override
    def name(self) -> str:
        return GRPO_GAME_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return GrpoGameFinetuneConfig
