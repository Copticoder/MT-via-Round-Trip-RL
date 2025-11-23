import os
import copy
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import wandb
import evaluate
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from lora_utils import apply_lora
from utils import (
    grpo_generate_sequences,
    grpo_compute_loss_and_logs,
)
from dl import TranslationDataModule
from vllm import LLM, SamplingParams

def extract_predicted_from_generated(generated, tokenizer, length_before_generation):
    output_ids = generated[length_before_generation:].tolist()
    if not output_ids:
        return ""

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content
def _run_evaluation(
    model: torch.nn.Module,
    tokenizer,
    dataloader,
    eval_cfg,
    *,
    device: torch.device,
    max_new_tokens: int,
    split_name: str = "eval",
    step_idx: int = 0,
):
    if eval_cfg is None or dataloader is None:
        return {}

    requested_metrics = getattr(eval_cfg, "translation_metric", [])
    if isinstance(requested_metrics, str):
        requested_metrics = [requested_metrics]
    requested_metrics = [metric.lower() for metric in requested_metrics]

    was_training = model.training
    model.eval()

    metric_prefix = split_name.strip().replace(" ", "_") or "eval"
    loader = dataloader
    predictions = []
    references = []
    sample_records = []

    # Progress bar setup
    try:
        total_batches = len(loader)
    except TypeError:
        # Fallback if len(val_loader) is not available
        total_batches = None

    with torch.no_grad():
        with tqdm(total=total_batches, desc=f"Evaluating ({metric_prefix})", leave=False) as pbar:
            for idx, batch in enumerate(loader):
                encoder_inputs, ground_truths, sample_ids = batch
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                else:
                    ground_truths = list(ground_truths)
                sample_ids = [int(sid) for sid in (sample_ids if isinstance(sample_ids, (list, tuple)) else [sample_ids])]
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in encoder_inputs.items()}
                generated = model.generate(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                )
                if generated.dim() == 1:
                    generated = generated.unsqueeze(0)
                for sample_idx, reference in enumerate(ground_truths):
                    # get the length of the input ids for this example before generation 
                    length_before_generation = encoder_inputs["input_ids"][sample_idx].size(0)
                    hypothesis = extract_predicted_from_generated(generated[sample_idx], tokenizer, length_before_generation, sample_idx)
                    ref_text = reference if isinstance(reference, str) else str(reference)
                    predictions.append(hypothesis)
                    references.append(ref_text)
                    sample_records.append((ref_text, hypothesis, sample_ids[sample_idx]))
                if pbar.total is not None:
                    pbar.update(1)

    results = {}
    if predictions and "bleu" in requested_metrics:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_res = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        results[f"{metric_prefix}/bleu"] = float(bleu_res.get("score", 0.0))
    if predictions and "chrf++" in requested_metrics:
        chrf_metric = CHRF(word_order=2, char_order=6)
        # sacrebleu expects references as list of reference sets
        score = chrf_metric.corpus_score(hypotheses=predictions, references=[references]).score
        results[f"{metric_prefix}/chrf++"] = float(score)

    if results:
        print(f"{split_name.capitalize()} evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

    if wandb.run is not None and (results or sample_records):
        log_payload = results.copy()
        # if sample_records:
            
        #     eval_table = wandb.Table(columns=["reference", "generated", "sample_id", "step"], log_mode="MUTABLE")
        #     # pick the same 50 samples for the eval table and set which step they are from
        #     sample_records = sample_records[:50]
        #     for reference, hypothesis, sample_id in sample_records:
        #         eval_table.add_data(reference, hypothesis, sample_id, step_idx)
        #     log_payload["eval/Translations"] = eval_table
        wandb.log(log_payload)

    if was_training:
        model.train()

    return results

@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    torch.manual_seed(config.seed)
    # Device mapping: policy (NLLB) on cuda:1, reference+goldfish on cuda:0
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        aux_device = torch.device("cuda:0")
        policy_device = torch.device("cuda:1")
    else:
        # Fallback to single GPU/CPU
        aux_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_device = aux_device
    # Build model and tokenizer
    model, tokenizer = get_model(config)
    model.to(policy_device)
    # Perplexity model for rewarding the generated sequences
    added_for_rus = "_1000mb"
    if config.task.data.target_lang in ["ayr_Latn", "bho_Deva", "dyu_Latn", "fur_Latn", "wol_Latn"]:
        added_for_rus = "_full"
    perplexity_config = AutoConfig.from_pretrained(f"goldfish-models/{config.task.data.target_lang}{added_for_rus}")
    perplexity_model = AutoModelForCausalLM.from_pretrained(
        f"goldfish-models/{config.task.data.target_lang}{added_for_rus}", config=perplexity_config
    )
    perplexity_tokenizer = AutoTokenizer.from_pretrained(
        f"goldfish-models/{config.task.data.target_lang}{added_for_rus}"
    )
    perplexity_model.to(aux_device)
    source_lang_code = config.task.data.source_lang
    target_lang_code = config.task.data.target_lang
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang_code
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = target_lang_code

    def _tokenize_with_lang(texts, src_lang):
        if isinstance(texts, str):
            texts = [texts]
        previous_src_lang = getattr(tokenizer, "src_lang", None)
        if previous_src_lang is not None:
            tokenizer.src_lang = src_lang
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if previous_src_lang is not None:
            tokenizer.src_lang = previous_src_lang
        return encoded

    eval_cfg = getattr(config.task, "eval", None)
    eval_only = bool(getattr(eval_cfg, "only", False)) if eval_cfg is not None else False
    run_eval = bool(getattr(eval_cfg, "run", False)) if eval_cfg is not None else False
    if eval_only:
        run_eval = True
    run_training = not eval_only
    eval_every_n_opt_steps = 0
    if run_eval and eval_cfg is not None:
        eval_every_n_opt_steps = int(getattr(eval_cfg, "every_n_opt_steps", 0))
    
    test_cfg = getattr(config.task, "test", None)
    run_test = bool(getattr(test_cfg, "run", False)) if test_cfg is not None else False
    test_eval_cfg = None
    if run_test:
        if eval_cfg is not None:
            test_eval_cfg = OmegaConf.merge(eval_cfg, test_cfg)
        else:
            test_eval_cfg = test_cfg

    # Initialize Weights & Biases
    model_name_for_run = str(getattr(config.task.model, "name", "model")).replace("/", "-")
    run_name = f"grpo_{model_name_for_run}_outcome_reward_batch_{config.task.training.batch_size}_src_tgt_src_grad_accum_self-supervised_rus"
    run_wandb = bool(getattr(config.task.training, "use_wandb", True))
    if run_wandb:
        wandb.init(
            project="grpo-translation-nllb-multi-domain",
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            dir="/root/wandb",
        )
    train_table = None
    if run_training and run_wandb:
        train_table = wandb.Table(columns=["reference", "generated", "sample_id"], log_mode="MUTABLE")
    # Reference model (frozen copy of the policy)
    reference_name = getattr(config.task.model, "reference_name", None)
    ref_model = None
    if run_training:
        if reference_name is not None:
            ref_model = AutoModelForSeq2SeqLM.from_pretrained(
                reference_name,
                attn_implementation="flash_attention_2",
                dtype=model.dtype,
            ).to(aux_device)
        else:
            ref_model = copy.deepcopy(model).to(aux_device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Data module
    illegal_token_mask = None
    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        sort_by_length=False,
        train_batch_size=int(getattr(config.task.training, "batch_size", 1)),
    )
    data.setup("fit")

    val_dataloader = data.val_dataloader() if run_eval else None
    test_dataloader = None
    if run_test:
        test_dataloader = data.test_dataloader()
        if test_dataloader is None:
            print("Test split not available; skipping test evaluation.")
            run_test = False
            test_eval_cfg = None

    # Training hyperparameters
    max_epochs = int(config.task.training.epochs)
    updates_per_batch = int(getattr(config.task.training, "updates_per_batch", 50))
    grad_accum_steps = int(getattr(config.task.training, "accumulate_grad_batches", 1))
    if grad_accum_steps < 1:
        raise ValueError("accumulate_grad_batches must be >= 1.")
    num_return_sequences = int(getattr(config.task.training, "num_return_sequences", 4))
    max_new_tokens = int(getattr(config.task.constraints, "max_sentence_len", 128))
    gen_temperature = float(getattr(config.task.training, "gen_temperature", 1.3))
    beta = float(getattr(config.task.training, "beta", 0.04))
    clip_param = float(getattr(config.task.training, "clip_param", 0.2))
    tgt_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    src_lang_id = tokenizer.convert_tokens_to_ids(source_lang_code)

    # Optimizer setup
    if run_training:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for the optimizer.")
        optimizer = torch.optim.AdamW(
            trainable_params, lr=float(getattr(config.task.training, "lr", 7e-6))
        )
        optimizer.zero_grad()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(
            f"Trainable parameters: {trainable_params_count:,} "
            f"({trainable_params_count / max(total_params, 1):.2%}) out of {total_params:,}"
        )
        accum_steps_since_update = 0
        optimizer_step = 0
        log_every_n_steps = max(int(getattr(config.task.training, "log_every_n_steps", 10)), 1)
    if run_eval:
        _run_evaluation(
            model,
            tokenizer,
            val_dataloader,
            eval_cfg,
            device=policy_device,
            max_new_tokens=max_new_tokens,
            split_name="eval",
        )

    if run_training:
        for epoch in range(max_epochs):
            train_loader = data.train_dataloader()
            step_idx = 0

            for batch in train_loader:
                src_prompt, ground_truths, sample_ids = batch
                encoder_inputs = {k: v.to(policy_device, non_blocking=True) for k, v in src_prompt.items()}
                batch_size = encoder_inputs["input_ids"].size(0)                
                while optimizer_step < updates_per_batch:
                    # Generate candidate sequences with current policy
                    generated_local = grpo_generate_sequences(
                        model,
                        encoder_inputs,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 100)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        
                    )
                    
                    seq_len = generated_local.size(1)
                    try:
                        generated_all = generated_local.reshape(batch_size, num_return_sequences, seq_len)
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to reshape generated sequences into (batch_size, num_return_sequences, seq_len). "
                            f"Batch size={batch_size}, num_return_sequences={num_return_sequences}, seq_len={seq_len}."
                        ) from exc
                     # Compute GRPO loss and update model
                    loss_forward, logs_forward = grpo_compute_loss_and_logs(
                        model,
                        ref_model,
                        tokenizer,
                        encoder_inputs,
                        generated_all,
                        ground_truths,
                        end_of_sentence_token_id=  151643, # <|endoftext|> token id for qwen3
                        beta=beta,
                        clip_param=clip_param,
                        tgt_lang_id=tgt_lang_id,
                        length_penalty_weight=float(getattr(config.task.reward, "length_penalty_weight", 0.0)),
                        goldfish_model=perplexity_model,
                        goldfish_tokenizer=perplexity_tokenizer,
                        goldfish_reward_weight=float(getattr(config.task.reward, "goldfish_reward_weight", 0.5)),
                    )
                    
                    loss_scale = 1.0 / float(grad_accum_steps)
                    loss_forward = loss_forward * loss_scale
                    loss_forward.backward()
                    performed_optimizer_step = False
                    accum_steps_since_update += 1
                    if accum_steps_since_update >= grad_accum_steps:
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_steps_since_update = 0
                        optimizer_step += 1
                        performed_optimizer_step = True
                        if optimizer_step % int(getattr(config.task.training, "update_ref_policy_every_n_steps", 16)) == 0:
                            # Refresh reference model at epoch boundaries
                            ref_model = copy.deepcopy(model).to(aux_device)
                            ref_model.eval()
                            for p in ref_model.parameters():
                                p.requires_grad_(False)

                updates_per_batch += getattr(config.task.training, "updates_per_batch", 50)
                if run_eval and eval_every_n_opt_steps > 0 and (optimizer_step % eval_every_n_opt_steps == 0):
                    _run_evaluation(
                        model,
                        tokenizer,
                        val_dataloader,
                        eval_cfg,
                        tgt_lang_id=tgt_lang_id,
                        device=policy_device,
                        max_new_tokens=max_new_tokens,
                        split_name="eval",
                    )

                print(
                    f"[epoch {epoch}] step {step_idx} (opt {optimizer_step}) | "
                    f"f_loss={logs_forward['loss'].item():.4f} f_kl={logs_forward['kl'].item():.4f} f_reward={logs_forward['reward'].item():.4f} f_chrf={logs_forward['chrf'].item():.4f} | ppl={logs_forward['ppl'].item():.4f} |"
                )
                # Print the reference and one generated sequence for inspection
                best_candidates = []
                for idx, reference in enumerate(ground_truths):
                    decoded = tokenizer.decode(
                        generated_all[idx, 0], skip_special_tokens=True
                    )
                    best_candidates.append((reference, decoded, sample_ids[idx]))
                ref_text, gen_text, ref_sample_id = best_candidates[0]
                print(f"Reference[{ref_sample_id}]: {ref_text}")
                print(f"Generated[{ref_sample_id}]: {gen_text}")
                if run_wandb:
                    # Log to Weights & Biases
                    wandb.log(
                        {
                            "train/loss": float(logs_forward["loss"].item()),
                            "train/kl": float(logs_forward["kl"].item()),
                            "train/reward": float(logs_forward["reward"].item()),
                            "train/chrf": float(logs_forward["chrf"].item()),
                            "train/ppl": float(logs_forward["ppl"].item()),
                        }
                    )
                step_idx += 1
            if accum_steps_since_update > 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_steps_since_update = 0
                optimizer_step += 1
                if run_eval and eval_every_n_opt_steps > 0 and (optimizer_step % eval_every_n_opt_steps == 0):
                    _run_evaluation(
                        model,
                        tokenizer,
                        val_dataloader,
                        eval_cfg,
                        tgt_lang_id=tgt_lang_id,
                        device=policy_device,
                        max_new_tokens=max_new_tokens,
                        split_name="eval",
                    )

    if run_eval and not eval_only:
        _run_evaluation(
            model,
            tokenizer,
            val_dataloader,
            eval_cfg,
            tgt_lang_id=tgt_lang_id,
            device=policy_device,
            max_new_tokens=max_new_tokens,
            split_name="eval",
        )
        
    if run_test:
        _run_evaluation(
            model,
            tokenizer,
            test_dataloader,
            test_eval_cfg,
            tgt_lang_id=tgt_lang_id,
            device=policy_device,
            max_new_tokens=max_new_tokens,
            split_name="test",
        )

    # Save final model when training occurred
    if run_training:
        save_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    if wandb.run is not None:
        wandb.finish()




def get_model(config: DictConfig):
    model_cfg = config.task.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name
    )

    use_lora = bool(getattr(model_cfg, "use_lora", False))
    if use_lora:
        for param in model.parameters():
            param.requires_grad_(False)

        lora_cfg = getattr(model_cfg, "lora", {})
        target_modules = tuple(getattr(lora_cfg, "target_modules", ("q_proj", "v_proj")))
        if not target_modules:
            raise ValueError("LoRA target modules must be a non-empty sequence of module suffixes.")

        replaced_modules = apply_lora(
            model=model,
            target_modules=target_modules,
            r=int(getattr(lora_cfg, "r", 8)),
            lora_alpha=float(getattr(lora_cfg, "alpha", 32)),
            lora_dropout=float(getattr(lora_cfg, "dropout", 0.05)),
        )

        print(f"LoRA enabled: adapted {len(replaced_modules)} modules with rank={int(getattr(lora_cfg, 'r', 8))}.")

    return model, tokenizer


if __name__ == "__main__":
    train()
