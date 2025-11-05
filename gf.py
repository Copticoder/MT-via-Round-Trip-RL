from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


goldfish_model = "goldfish-models/ayr_latn_full"

MAX_SEQ_LEN = 64

# Return the log-probability of the target text given the input text.
def score_text(model, tokenizer, input_text, target_text):
    loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    # Prepare inputs.
    input_tokens = tokenizer([input_text], add_special_tokens=False)['input_ids'][0]
    target_tokens = tokenizer([target_text], add_special_tokens=False)['input_ids'][0]
    sequence_tokens = input_tokens + target_tokens
    # Prepend [CLS] to input sequence, to match training format.
    sequence_tokens.insert(0, tokenizer.cls_token_id)  # Start token.
    assert len(sequence_tokens) <= MAX_SEQ_LEN
    sequence_tokens = torch.tensor([sequence_tokens])
    sequence_tokens = sequence_tokens.to("cuda")
    # Run model.
    # input_ids shape: (n_examples=1, seq_length). Sequence tokens includes
    # start of sequence token.
    outputs = model(input_ids=sequence_tokens,
                    output_hidden_states=False, return_dict=True)
    # Logits shape: (n_examples=1, seq_len, vocab_size).
    logits = outputs['logits'].detach()
    del outputs
    # Labels are the ground truth next token for each index.
    labels = sequence_tokens[:, 1:]  # Shape: (n_examples=1, seq_len-1).
    # Next token probabilities ignored for last token.
    logits = logits[:, :-1, :]
    # To apply loss, logits should be shape: (n_examples=1, vocab_size, seq_len-1).
    logits = torch.transpose(logits, 1, 2)
    # Loss shape: (n_examples=1, seq_len-1).
    # These are negative log probabilities (natural log), corresponding to each
    # token in sequence_tokens excluding the start token.
    losses = loss(logits, labels).cpu()
    # Only consider for the targets, not inputs.
    losses = losses[0, len(input_tokens):]
    logprobs = -1.0 * losses
    # Log-probability of entire target text is the sum of token log-probs.
    summed_logprobs = torch.sum(logprobs, dim=-1).item()
    return summed_logprobs

@torch.no_grad()
def visualize_top_k_next_tokens(model, tokenizer, input_text, k: int = 10):
    model.eval()
    # Prepare input ids (prepend CLS to match training format)
    input_tokens = tokenizer([input_text], add_special_tokens=False)["input_ids"][0]
    sequence_tokens = list(input_tokens)
    sequence_tokens.insert(0, tokenizer.cls_token_id)
    assert len(sequence_tokens) <= MAX_SEQ_LEN
    input_ids = torch.tensor([sequence_tokens], device="cuda")

    outputs = model(input_ids=input_ids, output_hidden_states=False, return_dict=True)
    logits = outputs["logits"]  # (1, L, V)
    last_logits = logits[0, -1]  # (V,)
    probs = torch.softmax(last_logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, k)
    top_probs = top_probs.tolist()
    top_indices = top_indices.tolist()

    # Simple ASCII bar visualization
    max_bar = 40
    print("\nTop next-token probabilities:")
    for rank, (token_id, p) in enumerate(zip(top_indices, top_probs), start=1):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded.strip() == "":
            decoded = tokenizer.convert_ids_to_tokens(token_id)
        token_display = decoded.replace("\n", "\\n").replace("\t", "\\t")
        bar = "█" * max(1, int(p * max_bar))
        print(f"{rank:2d}. {token_display:<15} {p*100:6.2f}%  {bar}")


if __name__ == "__main__":
    config = AutoConfig.from_pretrained(goldfish_model)
    tokenizer = AutoTokenizer.from_pretrained(goldfish_model)
    model = AutoModelForCausalLM.from_pretrained(
            goldfish_model, config=config).to("cuda")

    input_text = 'Walikipuniw jutawa! Jutir urutak '
    target_text = 'utjtamti?'

    # Score text.
    # Note that this function prepends [CLS] to the input sequence, to
    # match training format.
    logprob = score_text(model, tokenizer, input_text, target_text)
    print(logprob)

    # Example visualization call
    visualize_top_k_next_tokens(model, tokenizer, input_text, k=10)