import math
from typing import Sequence

import torch
from sacrebleu.metrics import CHRF
from gf import score_text as gf_score_text

# =========================
# Distributed GRPO Utilities
# =========================

@torch.no_grad()
def grpo_generate_sequences(
    model,
    tokenizer,
    encoder_inputs,
    tgt_lang_id,
    *,
    max_new_tokens: int,
    gen_temperature: float,
    num_return_sequences: int,
    top_k: int = 100,
    top_p: float = 0.9,
    end_of_sentence_token_id: int = None,
):
    eos_id = (
        end_of_sentence_token_id
        if end_of_sentence_token_id is not None
        else tokenizer.eos_token_id
    )
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=gen_temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
        top_k=top_k,
        top_p=top_p,
    )
    gen = model.generate(
        input_ids=encoder_inputs["input_ids"],
        attention_mask=encoder_inputs.get("attention_mask", None),
        forced_bos_token_id=tgt_lang_id,
        **generation_kwargs,
    )
    return gen


def _gather_log_probs_from_logits_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def grpo_compute_decoder_per_token_logps(
    model,
    tokenizer,
    encoder_inputs,
    decoder_input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    # Repeat encoder inputs to match number of sequences
    base_batch_size = encoder_inputs["input_ids"].size(0)
    batch_multiplier = decoder_input_ids.size(0) // base_batch_size
    if batch_multiplier * base_batch_size != decoder_input_ids.size(0):
        raise ValueError(
            "decoder_input_ids size does not align with encoder batch size. "
            f"Got encoder batch {base_batch_size} and decoder batch {decoder_input_ids.size(0)}."
        )
    repeated_input_ids = encoder_inputs["input_ids"].repeat_interleave(batch_multiplier, dim=0)
    attention_mask = encoder_inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(batch_multiplier, dim=0)

    decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()

    outputs = model(
        input_ids=repeated_input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, L, V)
    per_token_logps = _gather_log_probs_from_logits_logits(logits, target_ids)  # (B, L)
    return per_token_logps


def grpo_compute_loss_and_logs(
    model,
    ref_model,
    tokenizer,
    encoder_inputs,
    generated_sequences: torch.Tensor,
    ground_truths: Sequence[str],
    *,
    end_of_sentence_token_id: int,
    beta: float,
    clip_param: float,
    tgt_lang_id: int,
    goldfish_model=None,
    goldfish_tokenizer=None,
    goldfish_max_seq_len: int = 64,
    goldfish_reward_weight: float = 0.5,
):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    else:
        ground_truths = list(ground_truths)

    if generated_sequences.dim() != 3:
        raise ValueError(
            "generated_sequences must be a 3D tensor of shape (batch_size, num_candidates, seq_len). "
            f"Received tensor with shape {tuple(generated_sequences.shape)}."
        )

    device = next(model.parameters()).device
    ref_device = next(ref_model.parameters()).device if ref_model is not None else device
    gold_device = (
        next(goldfish_model.parameters()).device
        if goldfish_model is not None
        else device
    )
    batch_size, num_candidates, seq_len = generated_sequences.size()
    if batch_size != len(ground_truths):
        raise ValueError(
            f"Number of ground truths ({len(ground_truths)}) does not match generated batch size ({batch_size})."
        )
    flat_sequences = generated_sequences.reshape(batch_size * num_candidates, seq_len)

    # Prepare decoder inputs/targets
    decoder_input_ids = flat_sequences[:, :-1]
    target_ids = flat_sequences[:, 1:]

    # Compute per-token logps under current and reference policies
    per_token_logps = grpo_compute_decoder_per_token_logps(
        model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
    )
    with torch.no_grad():
        # Move inputs to reference model device for computation
        enc_ref = {k: v.to(ref_device, non_blocking=True) for k, v in encoder_inputs.items()}
        dec_in_ref = decoder_input_ids.to(ref_device, non_blocking=True)
        tgt_ref = target_ids.to(ref_device, non_blocking=True)
        ref_per_token_logps = grpo_compute_decoder_per_token_logps(
            ref_model, tokenizer, enc_ref, dec_in_ref, tgt_ref
        ).to(device, non_blocking=True)

    # Completion mask to ignore pads and tokens after first EOS
    is_pad = target_ids == tokenizer.pad_token_id
    is_eos = target_ids == end_of_sentence_token_id
    is_lang_id = target_ids == tgt_lang_id
    eos_cumsum = is_eos.cumsum(dim=-1)
    after_eos = eos_cumsum >= 1
    completion_mask = (~is_pad) & (~after_eos) & (~is_lang_id)
    completion_mask = completion_mask.reshape(batch_size, num_candidates, -1)

    # Decode generated sequences without special tokens for reward computation
    flat_gen = generated_sequences.reshape(-1, generated_sequences.size(-1))
    generated_texts = tokenizer.batch_decode(flat_gen, skip_special_tokens=True)
    references = [
        ground_truths[idx // num_candidates]
        for idx in range(len(generated_texts))
    ]

    # Reward via goldfish prefix probability and chrF prefix scores
    goldfish_rewards = []
    chrf_prefix_rewards = []
    chrf_metric = CHRF(word_order=2, char_order=6)
    chrf_vals = []
    @torch.no_grad()
    def goldfish_prefix_prob(prefix_text: str, target_text: str) -> float:
        if goldfish_model is None or goldfish_tokenizer is None:
            return 0.0
        # Use gf.score_text: returns summed log-prob of target given input
        # We use empty input_text to score unconditional probability of the prefix
        logp = gf_score_text(goldfish_model, goldfish_tokenizer, prefix_text, target_text)
        return torch.tensor(logp)

    # Build rewards aligned to decoder target length
    target_len = target_ids.size(-1)
    for hyp, ref in zip(generated_texts, references):
        # For logging: final CHRF once per sample
        chrf_vals.append(chrf_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score / 100.0)
        # Tokenize hyp with NLLB tokenizer to define NLLB-prefix boundaries
        nllb_tokens = tokenizer.tokenize(hyp)
        # Build rewards length equal to target_len, using best-effort prefixes
        g_prefix_rewards = torch.zeros((target_len,), device=device)
        c_prefix_rewards = torch.zeros((target_len,), device=device)
        for t in range(1, target_len + 1):
            if t <= len(nllb_tokens):
                prefix_text = tokenizer.convert_tokens_to_string(nllb_tokens[:t])
                target_text = tokenizer.convert_tokens_to_string(nllb_tokens[t:t+1])
                if target_text == "":
                    continue
                # Goldfish prefix probability
                g_prefix_rewards[t - 1] = goldfish_prefix_prob(prefix_text, target_text)
                # chrF++ prefix score (0..1)
                c_prefix_rewards[t - 1] = chrf_metric.corpus_score(hypotheses=[prefix_text], references=[[ref]]).score / 100.0
            else:
                # If beyond actual tokens, repeat last value
                g_prefix_rewards[t - 1] = g_prefix_rewards[t - 2] if t > 1 else 0.0
                c_prefix_rewards[t - 1] = c_prefix_rewards[t - 2] if t > 1 else 0.0
        goldfish_rewards.append(g_prefix_rewards)
        chrf_prefix_rewards.append(c_prefix_rewards)

    goldfish_rewards = torch.stack(goldfish_rewards, dim=0).reshape(batch_size, num_candidates, -1)
    chrf_prefix_rewards = torch.stack(chrf_prefix_rewards, dim=0).reshape(batch_size, num_candidates, -1)
    # Combine rewards (weighting goldfish prob and chrF prefix)
    rewards = goldfish_reward_weight * goldfish_rewards + (1.0 - goldfish_reward_weight) * chrf_prefix_rewards
    chrf_mean = torch.tensor(sum(chrf_vals) / max(len(chrf_vals), 1), device=device)
    standardized_rewards = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-4)
    # in process reward based grpo, the process supervision calculates the advantage of each token as the sum of the normalized rewards from the following steps, i.e., ˆ𝐴𝑖,𝑡 = 𝑖𝑛𝑑𝑒𝑥(𝑗)≥𝑡 𝑟𝑖𝑛𝑑𝑒𝑥(𝑗)
    advantages = torch.zeros_like(rewards)
    for i in range(batch_size):
        for j in range(num_candidates):
            for t in range(rewards.size(2)):
                advantages[i, j, t] = standardized_rewards[i, j, t:].sum()
    # per_token_logps is (B*N, L), reshape naturally to (B, N, L)
    per_token_logps = per_token_logps.reshape(batch_size, num_candidates, -1)

    # PPO-style ratio (on-policy baseline trick)
    ratio = torch.exp(per_token_logps - per_token_logps.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    per_token_adv_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty versus reference policy
    ref_minus_pi = ref_per_token_logps.reshape(batch_size, num_candidates, -1) - per_token_logps
    per_token_kl = torch.exp(ref_minus_pi) - ref_minus_pi - 1.0

    per_token_obj = per_token_adv_loss - beta * per_token_kl
    per_token_obj = per_token_obj * completion_mask

    # Mean over valid tokens per sequence, then batch mean
    token_counts = completion_mask.sum(dim=-1).clamp_min(1)
    # per_token
    seq_losses = -per_token_obj.sum(dim=-1) / token_counts
    loss = seq_losses.mean()

    with torch.no_grad():
        kl_mean = (per_token_kl * completion_mask).sum() / token_counts.sum()

    logs = {
        "loss": loss.detach(),
        "kl": kl_mean.detach(),
        "reward": rewards.mean().detach(),
        "chrf": chrf_mean.detach(),
    }
    return loss, logs
