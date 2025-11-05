#!/usr/bin/env python3
"""Utility to compare Ayr Latn tokenization between NLLB and Goldfish models."""

import argparse
from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_GOLDFISH_MODEL = "goldfish-models/ayr_Latn_full"
DEFAULT_EXAMPLES = [
    "Walikipuniw jutawa! Jutir urutak utjtamti?",
    "Nayax suma qamaña munata.",
    "Jichhax kunsa lurapxañani?",
]


@dataclass
class TokenizationSummary:
    tokens: List[str]
    ids: List[int]
    spans: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare how NLLB and Goldfish tokenizers segment Ayr Latn text.",
    )
    parser.add_argument(
        "--nllb-model",
        default=DEFAULT_NLLB_MODEL,
        help="Hugging Face model name or path for the NLLB tokenizer.",
    )
    parser.add_argument(
        "--goldfish-model",
        default=DEFAULT_GOLDFISH_MODEL,
        help="Hugging Face model name or path for the Goldfish tokenizer.",
    )
    parser.add_argument(
        "--text",
        nargs="*",
        help="One or more text snippets to compare. Overrides defaults when provided.",
    )
    parser.add_argument(
        "--file",
        help="Optional path to a UTF-8 text file with one example per line.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of text examples to process.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include special tokens (e.g., BOS/EOS) in the comparison.",
    )
    return parser.parse_args()


def collect_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        for snippet in args.text:
            snippet = snippet.strip()
            if snippet:
                texts.append(snippet)
                if len(texts) >= args.max_examples:
                    return texts
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            for line in handle:
                if len(texts) >= args.max_examples:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        for example in DEFAULT_EXAMPLES:
            if len(texts) >= args.max_examples:
                break
            texts.append(example)
    return texts[: args.max_examples]


def tokenize_with_offsets(tokenizer, text: str, add_special_tokens: bool) -> TokenizationSummary:
    encode_kwargs = {"add_special_tokens": add_special_tokens}
    if getattr(tokenizer, "is_fast", False):
        encode_kwargs["return_offsets_mapping"] = True
    encoding = tokenizer(text, **encode_kwargs)
    input_ids = encoding["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    spans: List[str] = []
    if getattr(tokenizer, "is_fast", False) and encoding.encodings:
        offsets = encoding.encodings[0].offsets
        for start, end in offsets:
            if start is None or end is None or start >= end:
                spans.append("")
            else:
                spans.append(text[start:end])
    else:
        spans = [""] * len(tokens)

    return TokenizationSummary(tokens=tokens, ids=input_ids, spans=spans)


def sanitize_token(token: str) -> str:
    if token is None:
        return ""
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    return token


def truncate(text: str, width: int) -> str:
    if text is None:
        return ""
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def format_id(token_id) -> str:
    return str(token_id) if token_id is not None else "-"


def print_side_by_side(
    nllb_summary: TokenizationSummary,
    goldfish_summary: TokenizationSummary,
) -> None:
    max_len = max(len(nllb_summary.tokens), len(goldfish_summary.tokens))
    header = (
        f"{'pos':>3} | {'NLLB token':<22} | {'id':>8} | {'span':<18} || "
        f"{'Goldfish token':<22} | {'id':>8} | {'span':<18}"
    )
    print(header)
    print("-" * len(header))
    for idx in range(max_len):
        n_token = nllb_summary.tokens[idx] if idx < len(nllb_summary.tokens) else ""
        n_id = nllb_summary.ids[idx] if idx < len(nllb_summary.ids) else None
        n_span = nllb_summary.spans[idx] if idx < len(nllb_summary.spans) else ""

        g_token = goldfish_summary.tokens[idx] if idx < len(goldfish_summary.tokens) else ""
        g_id = goldfish_summary.ids[idx] if idx < len(goldfish_summary.ids) else None
        g_span = goldfish_summary.spans[idx] if idx < len(goldfish_summary.spans) else ""

        print(
            f"{idx:>3} | {sanitize_token(n_token):<22} | {format_id(n_id):>8} | {truncate(n_span, 18):<18} || "
            f"{sanitize_token(g_token):<22} | {format_id(g_id):>8} | {truncate(g_span, 18):<18}"
        )


def compare_tokenizers(args: argparse.Namespace) -> None:
    texts = collect_texts(args)
    print(f"Loading NLLB tokenizer: {args.nllb_model}")
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model)
    print(f"Loading Goldfish tokenizer: {args.goldfish_model}")
    goldfish_tokenizer = AutoTokenizer.from_pretrained(args.goldfish_model)

    print("\n=== Tokenizer metadata ===")
    print(f"NLLB vocab size: {getattr(nllb_tokenizer, 'vocab_size', 'unknown')}")
    print(f"Goldfish vocab size: {getattr(goldfish_tokenizer, 'vocab_size', 'unknown')}")
    print(f"Special tokens NLLB: {nllb_tokenizer.all_special_tokens}")
    print(f"Special tokens Goldfish: {goldfish_tokenizer.all_special_tokens}")

    for index, text in enumerate(texts, start=1):
        print("\n" + "=" * 80)
        print(f"Example {index}: {text}")
        n_summary = tokenize_with_offsets(nllb_tokenizer, text, args.add_special_tokens)
        g_summary = tokenize_with_offsets(goldfish_tokenizer, text, args.add_special_tokens)

        print(
            f"NLLB produced {len(n_summary.tokens)} tokens | "
            f"Goldfish produced {len(g_summary.tokens)} tokens"
        )
        print_side_by_side(n_summary, g_summary)


def main() -> None:
    args = parse_args()
    compare_tokenizers(args)


if __name__ == "__main__":
    main()

