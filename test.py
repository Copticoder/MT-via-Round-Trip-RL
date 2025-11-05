import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "ayr_Latn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 1.2           # Gumbel temperature
MAX_NEW_TOKENS = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def straight_through_decode(
    model,
    tokenizer,
    input_text,
    src_lang="fra_Latn",
    tgt_lang="eng_Latn",
    max_new_tokens=50,
    min_len=1,
    tau=1.0,
):
    
    """
    Simple token-by-token generation for NLLB, similar to your utils.py pattern.
    """
    
    # Tokenize input
    tokenizer.src_lang = src_lang
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Get language tokens
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    eos_token_id = tokenizer.eos_token_id
    
    # Initialize with EOS token followed by target language token (NLLB requirement)
    batch_size = input_ids.shape[0]
    generated_ids = torch.full((batch_size, 2), eos_token_id, dtype=torch.long, device=model.device)
    generated_ids[:, 1] = tgt_lang_id
    
    # Track active sequences
    active_seqs = torch.ones(batch_size, dtype=torch.bool, device=model.device)
    embed_matrix = model.get_input_embeddings().weight  # (V, D)

    # Cache for efficiency
    past_key_values = None
    encoder_outputs = None
    
    log_probs = []
    soft_distributions = []
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Prepare decoder input
            if i == 0:
                decoder_input_ids = generated_ids
            else:
                decoder_input_ids = generated_ids[:, -1:]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids if encoder_outputs is None else None,
                attention_mask=attention_mask if encoder_outputs is None else None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Cache for next iteration
            if encoder_outputs is None:
                encoder_outputs = outputs.encoder_last_hidden_state
            past_key_values = outputs.past_key_values
            
            # Get logits for last position
            logits = outputs.logits[:, -1, :]
            
            # Apply constraints (similar to your utils.py)
            modified_logits = logits.clone()
            
            # Minimum length constraint
            if i < min_len:
                modified_logits[:, eos_token_id] = -torch.inf
            
            # Apply temperature and sample
            prob = F.gumbel_softmax(modified_logits, tau=tau, hard=True, dim=-1)
            soft_distributions.append(prob)
            next_tokens = torch.argmax(prob, dim=-1)
            
            # Only update active sequences
            next_tokens = torch.where(
                active_seqs.unsqueeze(-1),
                next_tokens,
                eos_token_id
            )
            
            # Update generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update active sequences
            active_seqs = active_seqs & (next_tokens.squeeze(-1) != eos_token_id)
            
            # Stop if all sequences terminated
            if not active_seqs.any():
                break
    soft_sequences = torch.stack(soft_distributions, dim=1)  # (B, L, V)
    embeddings = torch.matmul(soft_sequences, embed_matrix)  # (B, L, D)
    return generated_ids, embeddings

sentence = "The government approved the vaccination program for children."

# English → Aymara (differentiable)
hard_tgt, soft_tgt_embeds = straight_through_decode(
    model,
    tokenizer,
    input_text=sentence,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
    max_new_tokens=MAX_NEW_TOKENS,
    tau=TAU,
)

tgt_mask = (hard_tgt != tokenizer.pad_token_id).long()
aymara_text = tokenizer.batch_decode(hard_tgt, skip_special_tokens=True)[0]
print("[Forward translation]", aymara_text)

# Feed soft embeddings back into encoder
encoder_cycle = model.get_encoder()(
    inputs_embeds=soft_tgt_embeds,
    attention_mask=tgt_mask[:, : soft_tgt_embeds.size(1)],
    return_dict=True,
)

source_tokens = tokenizer(
    sentence,
    return_tensors="pt",
).to(DEVICE)
decoder_labels = source_tokens["input_ids"][:, 1:].contiguous()
decoder_inputs = source_tokens["input_ids"][:, :-1].contiguous()
decoder_mask = (decoder_inputs != tokenizer.pad_token_id).long()

cycle_outputs = model.get_decoder()(
    input_ids=decoder_inputs,
    attention_mask=decoder_mask,
    encoder_hidden_states=encoder_cycle.last_hidden_state,
    encoder_attention_mask=tgt_mask[:, : soft_tgt_embeds.size(1)],
    return_dict=True,
)

logits = model.lm_head(cycle_outputs.last_hidden_state)
cycle_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    decoder_labels.view(-1),
    ignore_index=tokenizer.pad_token_id,
)
cycle_loss.backward()
print("[Cycle loss]", cycle_loss.item())