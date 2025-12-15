# train/train_t5_span_corrupt.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.config import T5Config
from utils.logging import setup_json_logging, log_metrics
from utils.helpers import set_seed
from models.encoder_decoder import T5ForPreTraining


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/t5_tiny.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/t5_span")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    return parser.parse_args()


class SpanCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def apply_span_corruption(
    token_ids,
    tokenizer,
    corruption_rate=0.15,
):
    """
    - Randomly mask tokens with prob p
    - Group contiguous masked tokens into spans
    - Replace each span with <extra_id_k> (left-to-right)
    - Target = <extra_id_0> span0 <extra_id_1> span1 ... <eos>
    """
    if len(token_ids) == 0:
        return token_ids, [tokenizer.eos_token_id]

    # Sample token-level mask
    mask = torch.bernoulli(
        torch.full((len(token_ids),), corruption_rate)
    ).bool().tolist()

    if not any(mask):
        return token_ids, [tokenizer.eos_token_id]

    # Build spans
    spans = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            span_tokens = []
            while i < len(mask) and mask[i]:
                span_tokens.append(token_ids[i])
                i += 1
            spans.append((start, span_tokens))
        else:
            i += 1

    # Build corrupted input + target
    input_ids = []
    target_ids = []
    prev_end = 0

    for span_idx, (start, span_tokens) in enumerate(spans):
        sentinel = tokenizer.convert_tokens_to_ids(f"<extra_id_{span_idx}>")

        input_ids.extend(token_ids[prev_end:start])
        input_ids.append(sentinel)

        target_ids.append(sentinel)
        target_ids.extend(span_tokens)

        prev_end = start + len(span_tokens)

    input_ids.extend(token_ids[prev_end:])
    target_ids.append(tokenizer.eos_token_id)

    return input_ids, target_ids


def collate_fn(batch, tokenizer, max_length=128, corruption_rate=0.15):
    # Tokenize raw texts
    tokenized = tokenizer(
        batch,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    encoder_inputs = []
    decoder_inputs = []
    labels_list = []

    for text_ids in tokenized["input_ids"]:
        if not text_ids:
            text_ids = [tokenizer.pad_token_id]

        enc_ids, target = apply_span_corruption(text_ids, tokenizer, corruption_rate)
        
        # Truncate
        enc_ids = enc_ids[:max_length]
        target = target[:max_length]
        
        # - Encoder input: corrupted sequence
        # - Decoder input: <pad> + target[:-1]
        # - Labels: target (loss ignores padding via -100)
        encoder_inputs.append(enc_ids)
        decoder_inputs.append([tokenizer.pad_token_id] + target[:-1])
        labels_list.append(target)

    # Pad all sequences
    def pad_and_mask(seqs, pad_value):
        max_len = max(len(s) for s in seqs)
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append(s + [pad_value] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    input_ids, attention_mask = pad_and_mask(encoder_inputs, tokenizer.pad_token_id)
    decoder_input_ids, decoder_attention_mask = pad_and_mask(decoder_inputs, tokenizer.pad_token_id)
    
    # -100 to be ignored in loss
    labels, _ = pad_and_mask(labels_list, -100)

    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "encoder_attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }

class CollateFn:
    def __init__(self, tokenizer, max_length, corruption_rate, mean_span_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length

    def __call__(self, batch):
        return collate_fn(
            batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            corruption_rate=self.corruption_rate,
        )


@torch.no_grad()
def evaluate(model, dataloader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        dec_input = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        enc_mask = batch["encoder_attention_mask"].to(device)
        dec_mask = batch["decoder_attention_mask"].to(device)

        logits = model(input_ids, dec_input, enc_mask, dec_mask)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        # Count non-ignored tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    model.train()
    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    config = T5Config.from_yaml(args.config)

    logger = setup_json_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        project_name="t5-span-tiny",
        config=config.to_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id

    collate_fn_obj = CollateFn(
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        corruption_rate=config.span_corruption_rate,
        mean_span_length=config.mean_span_length,
    )

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:50000]

    train_dataset = SpanCorruptionDataset(texts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_obj,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForPreTraining(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler()

    model.train()
    global_step = 0
    running_loss = 0.0

    print(f"Starting training for {args.num_steps} steps...")
    while global_step < args.num_steps:
        for batch in train_loader:
            if global_step >= args.num_steps:
                break

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    decoder_input_ids=batch["decoder_input_ids"].to(device),
                    encoder_attention_mask=batch["encoder_attention_mask"].to(device),
                    decoder_attention_mask=batch["decoder_attention_mask"].to(device),
                )

                labels = batch["labels"].to(device)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, config.vocab_size), labels.view(-1)
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum_steps
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(f"Step {global_step} | Span Loss: {avg_loss:.4f}")
                log_metrics({"train/span_loss": avg_loss, "step": global_step})
                running_loss = 0.0

            if global_step % args.eval_every == 0:
                val_texts = ds["text"][50000:51000]
                val_dataset = SpanCorruptionDataset(val_texts)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn_obj,
                )
                val_loss = evaluate(model, val_loader, device, config.vocab_size)
                print(f"Step {global_step} | Val Loss: {val_loss:.4f}")
                log_metrics({"val/span_loss": val_loss, "step": global_step})

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()