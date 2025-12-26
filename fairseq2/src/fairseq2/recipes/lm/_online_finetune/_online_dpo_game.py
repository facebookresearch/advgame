# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Final, Tuple, final

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
from fairseq2.recipes.lm._instruction_finetune import update_nll_loss
from fairseq2.recipes.lm._online_finetune._common import (
    ScoredRolloutTree,
    ScoredRolloutTreeDirect,
    StatefulRolloutBagGame,
    VllmGameSyncSection,
    _collate,
    compute_token_level_entropy,
    update_avg_dpo_grad_factor,
    update_avg_loss_zeroer,
    update_batch_metrics,
    update_dpo_loss,
    update_logit_entropy,
    xError,
    yError,
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_model import (
    RemoteVllmModel,
    maybe_sync_model,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardGameSection,
    VLLMOutputRewardGame,
    VLLMOutputRewardGameHandler,
)
from fairseq2.recipes.lm._preference_finetune._common import (
    _gather_lprobs_avg,
    update_logps_metrics,
    update_sequence_length_metrics,
)
from fairseq2.utils.list import (
    largmax_filtered,
    largmin_filtered,
    linit,
)
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import ValidationError, ValidationResult, validate


@final
class OnlineDpoGameFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model game DPO-finetuning unit with online generations.
    Based on the online DPO paper: https://arxiv.org/abs/2305.18290."""

    _step_nr: int
    _valid_step_nr: int
    _model_attacker: Model
    _model_defender: Model
    _ema_model_attacker: Model
    _ema_model_defender: Model
    _ref_model_attacker: Model
    _ref_model_defender: Model
    _config: OnlineDpoGameFinetuneConfig
    _model_update_group: PyNcclCommunicator
    _display_name: str
    _reward: VLLMOutputRewardGame

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
        config: OnlineDpoGameFinetuneConfig,
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

        self._display_name = "online_dpo_game"

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
            f"[{prefix}] Starting AdvGameDPO call at step nr: {self._step_nr} with batch size {prompt_batch.batch_size} and skip_attacker={skip_attacker}"
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

        batch_attacker: PreferenceBatch | None
        batch_defender: PreferenceBatch
        (batch_attacker, batch_defender, bad_batch_attacker, bad_batch_defender) = (
            self._prepare_preference_batch(srt, metric_bag=metric_bag, prefix=prefix)
        )

        if batch_attacker is None:
            loss_attacker = torch.tensor(0.0)
            batch_size_attacker = 0
        else:
            loss_attacker = self._compute_loss(
                batch=batch_attacker,
                model=self._model_attacker,
                ref_model=self._ref_model_attacker,
                ema_model=self._ema_model_attacker,
                prefix=prefix + "attacker/",
                is_bad_batch=bad_batch_attacker,
                metric_bag=metric_bag,
            )
            batch_size_attacker = batch_attacker.chosen.batch_size

        loss_defender = self._compute_loss(
            batch=batch_defender,
            model=self._model_defender,
            ref_model=self._ref_model_defender,
            ema_model=self._ema_model_defender,
            prefix=prefix + "defender/",
            is_bad_batch=bad_batch_defender,
            metric_bag=metric_bag,
        )
        batch_size_defender = batch_defender.chosen.batch_size

        log.info(
            f"[{prefix}] Finished AdvGameDPO call. loss_attacker={loss_attacker.item()}, loss_defender={loss_defender.item()}, batch_size_attacker={batch_size_attacker}, batch_size_defender={batch_size_defender}"
        )
        return loss_attacker, loss_defender, batch_size_attacker, batch_size_defender

    def _compute_loss(
        self,
        batch: PreferenceBatch,
        model: Model,
        ref_model: Model,
        ema_model: Model,
        prefix: str,
        is_bad_batch: bool,
        metric_bag: MetricBag,
    ) -> Tensor:
        loss_zeroer = 0.0 if is_bad_batch else 1.0

        # TODO: this throws error, fix it
        # if is_bad_batch:
        #     metric_bag.get(Sum, prefix + "num_examples").update(int(0))
        # else:
        #     metric_bag.get(Sum, prefix + "num_examples").update(batch.batch_size)
        update_sequence_length_metrics(
            metric_bag, batch, weight_factor=loss_zeroer, prefix=prefix
        )

        (
            chosen_logps,
            rejected_logps,
            avg_chosen_logps,
            avg_rejected_logps,
            tgt_logit_entropy,
            nll_loss,
        ) = self._process_batch(
            model=model,
            batch=batch,
            metric_bag=metric_bag,
            prefix=prefix + "model/",
            loss_zeroer=loss_zeroer,
            compute_nll=True,
        )

        with torch.no_grad():
            (
                ref_chosen_logps,
                ref_rejected_logps,
                ref_avg_chosen_logps,
                ref_avg_rejected_logps,
                _,
                _,
            ) = self._process_batch(
                model=ref_model,
                batch=batch,
                metric_bag=metric_bag,
                loss_zeroer=loss_zeroer,
                prefix=prefix + "ref_model/",
            )
            if (
                self._config.loss_config.log_ema_model_metrics
                or self._config.loss_config.use_ema_model_as_ref_model
            ):
                (
                    ema_chosen_logps,
                    ema_rejected_logps,
                    ema_avg_chosen_logps,
                    ema_avg_rejected_logps,
                    _,
                    _,
                ) = self._process_batch(
                    model=ema_model,
                    batch=batch,
                    metric_bag=metric_bag,
                    loss_zeroer=loss_zeroer,
                    prefix=prefix + "ema_model/",
                )

        if self._config.loss_config.use_ema_model_as_ref_model:
            ref_chosen_logps = ema_chosen_logps
            ref_rejected_logps = ema_rejected_logps
            ref_avg_chosen_logps = ema_avg_chosen_logps
            ref_avg_rejected_logps = ema_avg_rejected_logps

        if self._config.loss_config.length_normalization:
            _, _, dpo_loss, avg_dpo_grad_factor = self._compute_dpo_ipo_loss(
                avg_chosen_logps,
                ref_avg_chosen_logps,
                avg_rejected_logps,
                ref_avg_rejected_logps,
            )
        else:
            _, _, dpo_loss, avg_dpo_grad_factor = self._compute_dpo_ipo_loss(
                chosen_logps,
                ref_chosen_logps,
                rejected_logps,
                ref_rejected_logps,
            )
        update_dpo_loss(
            metric_bag,
            dpo_loss,
            batch_size=batch.chosen.batch_size,
            weight_factor=loss_zeroer,
            prefix=prefix,
        )
        update_avg_dpo_grad_factor(
            metric_bag,
            avg_dpo_grad_factor,
            batch_size=batch.chosen.batch_size,
            weight_factor=loss_zeroer,
            prefix=prefix,
        )

        if self._config.loss_config.nll_length_normalization:
            _, chosen_target_batch = batch.chosen.as_auto_regressive()
            nll_loss = (
                nll_loss
                * chosen_target_batch.batch_size
                / chosen_target_batch.num_target_elements
            )
        update_nll_loss(
            metric_bag,
            nll_loss,
            batch.chosen.num_target_elements * loss_zeroer,
            prefix=prefix,
        )

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum()
            * self._config.loss_config.entropy_regularizer_scale
        )
        loss = (
            dpo_loss
            + self._config.loss_config.nll_scale * nll_loss
            + max_entropy_regularizer
        )  # nll normalization applied locally per-rank

        update_avg_loss_zeroer(metric_bag, torch.tensor(loss_zeroer), prefix=prefix)
        loss = loss * loss_zeroer  # zero loss if entire batch was dummy batch

        return loss

    def _process_batch(
        self,
        model: Model,
        batch: PreferenceBatch,
        metric_bag: MetricBag,
        prefix: str,
        loss_zeroer: float,
        compute_nll: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        chosen_input_batch, chosen_target_batch = batch.chosen.as_auto_regressive()
        (
            chosen_input_batch_seqs,
            chosen_input_batch_layout,
        ) = chosen_input_batch.as_input()

        rejected_input_batch, rejected_target_batch = (
            batch.rejected.as_auto_regressive()
        )
        (
            rejected_input_batch_seqs,
            rejected_input_batch_layout,
        ) = rejected_input_batch.as_input()

        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        if compute_nll:
            nll_loss, chosen_logits = model.module(
                chosen_input_batch_seqs,
                chosen_input_batch_layout,
                targets=chosen_target_batch.seqs,
                target_mask=chosen_target_batch.target_mask,
                return_logits=True,
            )
        else:
            nll_loss = None
            chosen_logits = model.module(
                chosen_input_batch_seqs, chosen_input_batch_layout
            )
        rejected_logits = model.module(
            rejected_input_batch_seqs, rejected_input_batch_layout
        )

        chosen_logps, avg_chosen_logps = _gather_lprobs_avg(
            chosen_logits, chosen_target_batch
        )
        rejected_logps, avg_rejected_logps = _gather_lprobs_avg(
            rejected_logits, rejected_target_batch
        )
        update_logps_metrics(
            metric_bag,
            batch,
            chosen_logps,
            rejected_logps,
            avg_chosen_logps,
            avg_rejected_logps,
            weight_factor=loss_zeroer,
            prefix=prefix,
        )

        tgt_logit_entropy = compute_token_level_entropy(
            chosen_logits, chosen_target_batch.target_mask
        )  # [Batch x Rollouts, 1]
        update_logit_entropy(metric_bag, tgt_logit_entropy, prefix=prefix)

        return (
            chosen_logps,
            rejected_logps,
            avg_chosen_logps,
            avg_rejected_logps,
            tgt_logit_entropy,
            nll_loss,
        )

    def _compute_dpo_ipo_loss(
        self,
        chosen_xy_logps: Tensor,
        ref_chosen_xy_logps: Tensor,
        rejected_xy_logps: Tensor,
        ref_rejected_xy_logps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._config.loss_config.loss_type == "dpo":
            return self._compute_dpo_loss(
                chosen_xy_logps,
                ref_chosen_xy_logps,
                rejected_xy_logps,
                ref_rejected_xy_logps,
            )
        elif self._config.loss_config.loss_type == "ipo":
            return self._compute_ipo_loss(
                chosen_xy_logps,
                ref_chosen_xy_logps,
                rejected_xy_logps,
                ref_rejected_xy_logps,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self._config.loss_config.loss_type}. Supported types are 'dpo' and 'ipo'."
            )

    def _compute_dpo_loss(
        self,
        chosen_xy_logps: Tensor,
        ref_chosen_xy_logps: Tensor,
        rejected_xy_logps: Tensor,
        ref_rejected_xy_logps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logp_ratio_chosen_xy = self._config.loss_config.beta * (
            chosen_xy_logps - ref_chosen_xy_logps
        )
        logp_ratio_rejected_xy = self._config.loss_config.beta * (
            rejected_xy_logps - ref_rejected_xy_logps
        )
        dpo_loss = -torch.nn.functional.logsigmoid(
            logp_ratio_chosen_xy - logp_ratio_rejected_xy
        )
        dpo_grad_factor = torch.nn.functional.sigmoid(
            logp_ratio_rejected_xy - logp_ratio_chosen_xy
        )
        avg_dpo_grad_factor = dpo_grad_factor.mean()
        return (
            logp_ratio_chosen_xy,
            logp_ratio_rejected_xy,
            dpo_loss.sum(),
            avg_dpo_grad_factor,
        )

    def _compute_ipo_loss(
        self,
        chosen_xy_logps: Tensor,
        ref_chosen_xy_logps: Tensor,
        rejected_xy_logps: Tensor,
        ref_rejected_xy_logps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logp_ratio_chosen_xy = chosen_xy_logps - ref_chosen_xy_logps
        logp_ratio_rejected_xy = rejected_xy_logps - ref_rejected_xy_logps
        ipo_loss = (
            logp_ratio_chosen_xy
            - logp_ratio_rejected_xy
            - 1 / (2 * self._config.loss_config.beta)
        ) ** 2
        avg_ipo_grad_factor = torch.tensor(1.0)
        return (
            logp_ratio_chosen_xy,
            logp_ratio_rejected_xy,
            ipo_loss.sum(),
            avg_ipo_grad_factor,
        )

    def _prepare_preference_batch(
        self,
        srt: ScoredRolloutTree | ScoredRolloutTreeDirect,
        metric_bag: MetricBag,
        prefix: str,
    ) -> Tuple[PreferenceBatch, PreferenceBatch, bool, bool]:
        if isinstance(srt, ScoredRolloutTreeDirect):
            batch_xy, bad_batch_xy = self._prepare_preference_batch_direct(
                srtd=srt, metric_bag=metric_bag, prefix=prefix
            )
            return None, batch_xy, True, bad_batch_xy

        log.info(f"[{prefix}] Preparing preference batches from scored rollout tree")
        B, A = (srt.B, srt.A)

        d_chosen_BA = linit([B, A])
        d_rejected_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            if srt.x_error_BA[b][a] == xError.NO_ERROR:
                d_chosen = largmax_filtered(srt.y_reward_patched_BAD[b][a])
                d_rejected = largmin_filtered(srt.y_reward_patched_BAD[b][a])
                d_chosen_BA[b][a] = d_chosen
                d_rejected_BA[b][a] = d_rejected

        log.info(f"[{prefix}] Chosen defender indices: {d_chosen_BA}")
        log.info(f"[{prefix}] Rejected defender indices: {d_rejected_BA}")

        a_chosen_B = linit([B])
        a_rejected_B = linit([B])
        for b in range(B):
            a_chosen = largmax_filtered(srt.x_reward_patched_BA[b])
            a_rejected = largmin_filtered(srt.x_reward_patched_BA[b])
            a_chosen_B[b] = a_chosen
            a_rejected_B[b] = a_rejected

        log.info(f"[{prefix}] Chosen attacker indices: {a_chosen_B}")
        log.info(f"[{prefix}] Rejected attacker indices: {a_rejected_B}")

        chosen_sx_L = []
        rejected_sx_L = []
        s_emb_lens_L = []

        chosen_xy_K = []
        rejected_xy_K = []
        x_emb_chosen_lens_K = []
        x_emb_rejected_lens_K = []

        for b in range(B):
            a_chosen = a_chosen_B[b]
            a_rejected = a_rejected_B[b]

            if a_chosen is not None and a_rejected is not None:
                chosen_reward_att = srt.x_reward_patched_BA[b][a_chosen]
                rejected_reward_att = srt.x_reward_patched_BA[b][a_rejected]

                if chosen_reward_att != rejected_reward_att:
                    x_raw_tokens_chosen = srt.x_raw_tokens_BA[b][a_chosen]
                    x_raw_tokens_rejected = srt.x_raw_tokens_BA[b][a_rejected]

                    if (
                        srt.x_error_BA[b][a_rejected] == xError.MAX_LENGTH
                        and self._config.loss_config.clip_rejected_length_on_max_length_error
                        and len(x_raw_tokens_chosen) < len(x_raw_tokens_rejected)
                    ):
                        clip_len = len(x_raw_tokens_chosen)
                        x_raw_tokens_rejected = x_raw_tokens_rejected[:clip_len]

                    chosen_sx_L.append(srt.s_emb_tokens_B[b] + x_raw_tokens_chosen)
                    rejected_sx_L.append(srt.s_emb_tokens_B[b] + x_raw_tokens_rejected)
                    s_emb_lens_L.append(len(srt.s_emb_tokens_B[b]))

            for a in range(A):
                d_chosen = d_chosen_BA[b][a]
                d_rejected = d_rejected_BA[b][a]

                if d_chosen is not None and d_rejected is not None:
                    chosen_reward_def = srt.y_reward_patched_BAD[b][a][d_chosen]
                    rejected_reward_def = srt.y_reward_patched_BAD[b][a][d_rejected]

                    if chosen_reward_def != rejected_reward_def:
                        y_raw_tokens_chosen = srt.y_raw_tokens_BAD[b][a][d_chosen]
                        y_raw_tokens_rejected = srt.y_raw_tokens_BAD[b][a][d_rejected]

                        if (
                            srt.y_error_BAD[b][a][d_rejected] == yError.MAX_LENGTH
                            and self._config.loss_config.clip_rejected_length_on_max_length_error
                            and len(y_raw_tokens_chosen) < len(y_raw_tokens_rejected)
                        ):
                            clip_len = len(y_raw_tokens_chosen)
                            y_raw_tokens_rejected = y_raw_tokens_rejected[:clip_len]

                        chosen_xy_K.append(
                            srt.x_emb_tokens_BA[b][a] + y_raw_tokens_chosen
                        )
                        rejected_xy_K.append(
                            srt.x_emb_tokens_BA[b][a] + y_raw_tokens_rejected
                        )
                        x_emb_chosen_lens_K.append(len(srt.x_emb_tokens_BA[b][a]))
                        x_emb_rejected_lens_K.append(len(srt.x_emb_tokens_BA[b][a]))

        # TODO: more elegant way to do this?
        dummy_chosen_tokens = [0] * 10
        dummy_rejected_tokens = [0] * 10
        dummy_prompt_length = 5

        if len(chosen_sx_L) == 0:
            bad_batch_sx = True
            chosen_sx_L = [dummy_chosen_tokens]
            rejected_sx_L = [dummy_rejected_tokens]
            s_emb_lens_L = [dummy_prompt_length]
        else:
            bad_batch_sx = False

        chosen_sx_L = _collate(chosen_sx_L, s_emb_lens_L, device=self._gangs.dp.device)
        rejected_sx_L = _collate(
            rejected_sx_L, s_emb_lens_L, device=self._gangs.dp.device
        )
        batch_sx = PreferenceBatch(
            chosen=chosen_sx_L,
            rejected=rejected_sx_L,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )

        if len(chosen_xy_K) == 0:
            bad_batch_xy = True
            chosen_xy_K = [dummy_chosen_tokens]
            rejected_xy_K = [dummy_rejected_tokens]
            x_emb_chosen_lens_K = [dummy_prompt_length]
            x_emb_rejected_lens_K = [dummy_prompt_length]
        else:
            bad_batch_xy = False

        chosen_xy_K = _collate(
            chosen_xy_K, x_emb_chosen_lens_K, device=self._gangs.dp.device
        )
        rejected_xy_K = _collate(
            rejected_xy_K, x_emb_rejected_lens_K, device=self._gangs.dp.device
        )
        batch_xy = PreferenceBatch(
            chosen=chosen_xy_K,
            rejected=rejected_xy_K,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )
        log.info(
            f"[{prefix}] Finished preparing preference batches. bad_batch_sx={bad_batch_sx}, bad_batch_xy={bad_batch_xy}"
        )
        return (batch_sx, batch_xy, bad_batch_sx, bad_batch_xy)

    def _prepare_preference_batch_direct(
        self, srtd: ScoredRolloutTreeDirect, metric_bag: MetricBag, prefix: str
    ) -> Tuple[PreferenceBatch, bool]:
        log.info(
            f"[{prefix}] Preparing direct preference batch from scored rollout tree"
        )
        B = srtd.B

        d_chosen_B = linit([B])
        d_rejected_B = linit([B])
        for b in range(B):
            d_chosen = largmax_filtered(srtd.y_reward_patched_BD[b])
            d_rejected = largmin_filtered(srtd.y_reward_patched_BD[b])
            d_chosen_B[b] = d_chosen
            d_rejected_B[b] = d_rejected

        log.info(f"[{prefix}] Chosen defender indices: {d_chosen_B}")
        log.info(f"[{prefix}] Rejected defender indices: {d_rejected_B}")

        chosen_sy_L = []
        rejected_sy_L = []
        s_emb_lens_L = []

        for b in range(B):
            d_chosen = d_chosen_B[b]
            d_rejected = d_rejected_B[b]

            if d_chosen is not None and d_rejected is not None:
                chosen_reward_def = srtd.y_reward_patched_BD[b][d_chosen]
                rejected_reward_def = srtd.y_reward_patched_BD[b][d_rejected]

                if chosen_reward_def != rejected_reward_def:
                    y_raw_tokens_chosen = srtd.y_raw_tokens_BD[b][d_chosen]
                    y_raw_tokens_rejected = srtd.y_raw_tokens_BD[b][d_rejected]

                    if (
                        srtd.y_error_BD[b][d_rejected] == yError.MAX_LENGTH
                        and self._config.loss_config.clip_rejected_length_on_max_length_error
                        and len(y_raw_tokens_chosen) < len(y_raw_tokens_rejected)
                    ):
                        clip_len = len(y_raw_tokens_chosen)
                        y_raw_tokens_rejected = y_raw_tokens_rejected[:clip_len]

                    chosen_sy_L.append(srtd.s_emb_tokens_B[b] + y_raw_tokens_chosen)
                    rejected_sy_L.append(srtd.s_emb_tokens_B[b] + y_raw_tokens_rejected)
                    s_emb_lens_L.append(len(srtd.s_emb_tokens_B[b]))

        dummy_chosen_tokens = [0] * 10
        dummy_rejected_tokens = [0] * 10
        dummy_prompt_length = 5

        if len(chosen_sy_L) == 0:
            bad_batch_sy = True
            chosen_sy_L = [dummy_chosen_tokens]
            rejected_sy_L = [dummy_rejected_tokens]
            s_emb_lens_L = [dummy_prompt_length]
        else:
            bad_batch_sy = False

        chosen_sy_L = _collate(chosen_sy_L, s_emb_lens_L, device=self._gangs.dp.device)
        rejected_sy_L = _collate(
            rejected_sy_L, s_emb_lens_L, device=self._gangs.dp.device
        )
        batch_sy = PreferenceBatch(
            chosen=chosen_sy_L,
            rejected=rejected_sy_L,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )
        log.info(
            f"[{prefix}] Finished preparing direct preference batch. bad_batch_sy={bad_batch_sy}"
        )
        return (batch_sy, bad_batch_sy)

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


ONLINE_DPO_GAME_FINETUNE_UNIT: Final = "online_dpo_game"


@dataclass(kw_only=True)
class DpoGameLossConfig:
    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    loss_type: str = "dpo"
    """The type of loss to use. Supported types are 'dpo' and 'ipo'."""

    nll_scale: float = 0.0
    nll_length_normalization: bool = True
    """The coefficient of NLL loss added to the DPO game loss."""

    length_normalization: bool = False
    """Use length normalized DPO game, which uses the average log probability of a sequence as the implicit reward."""

    entropy_regularizer_scale: float = 0.0

    log_rollouts: bool = False
    """Log rollouts during training/validation"""

    log_ema_model_metrics: bool = False
    """Log EMA model metrics if EMA model is provided."""

    use_ema_model_as_ref_model: bool = False
    """Use the EMA model as the reference model instead of the provided reference model."""

    clip_rejected_length_on_max_length_error: bool = False
    """Clip the rejected sequence to the chosen sequence length if the rejected sequence hits max length."""

    num_microbatches_B: int = 1
    num_microbatches_A: int = 1
    num_microbatches_D: int = 1
    """Number of mini-batches used over dimensions B, A, D for the rollout bag."""

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
class OnlineDpoGameFinetuneConfig:
    loss_config: DpoGameLossConfig = field(default_factory=lambda: DpoGameLossConfig())

    reward: RewardGameSection = field(
        default_factory=lambda: RewardGameSection(name="generative_game_verifier")
    )

    vllm_sync: VllmGameSyncSection = field(
        default_factory=lambda: VllmGameSyncSection()
    )


@final
class OnlineDpoGameFinetuneUnitHandler(OnlineFinetuneUnitHandler):
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
        vllm_actors: Dict[str, RemoteVllmModel],
    ) -> TrainUnit[PreferenceBatch]:
        config = structure(recipe_config.criterion.config, OnlineDpoGameFinetuneConfig)

        validate(config)
        log.info(f"DPO game loss config:\n{config}")

        reward_registry = self._context.get_registry(VLLMOutputRewardGameHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            vllm_actors=vllm_actors,
            reward_config=config.reward.config,
            gangs=gangs,
            context=self._context,
        )

        # TODO: add the force syncing back here, make sync mode passable or use force_sync?

        # TODO: decide converter as part of the model handler
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

        log.info("DPO game setup complete.")

        return OnlineDpoGameFinetuneUnit(
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
        return ONLINE_DPO_GAME_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OnlineDpoGameFinetuneConfig
