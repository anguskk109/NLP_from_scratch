# train/train_gpt_lm.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.config import GPTConfig
from utils.logging import setup_json_logging, log_metrics
from utils.helpers import set_seed
from models.decoder_only import GPTForPreTraining


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt_tiny.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/gpt_lm")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_gen_length", type=int, default=50)
    return parser.parse_args()


class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch, tokenizer, max_length=128):

    encoding = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,  # no [CLS]/[SEP] for GPT
    )
    return {
        "input_ids": encoding["input_ids"],
        "encoder_attention_mask": encoding["attention_mask"],
    }


class CollateFn:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        return collate_fn(batch, tokenizer=self.tokenizer, max_length=self.max_length)


@torch.no_grad()
def evaluate(model, dataloader, device, vocab_size, pad_token_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["encoder_attention_mask"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)

        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # Count non-padded tokens for accurate perplexity
        num_tokens = shift_attention_mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return avg_loss, perplexity


def top_k_filtering(logits, top_k):
    if top_k <= 0:
        return logits

    values, _ = torch.topk(logits, top_k)
    min_values = values[..., -1, None]
    logits = torch.where(logits < min_values, -float("Inf"), logits)
    return logits

def top_p_filtering(logits, top_p):
    if top_p <= 0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    # Mask tokens with cumulative prob above threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # clone to avoid overlapping memory write
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    return logits


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_length,
    device,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["encoder_attention_mask"].to(device)

    for _ in range(max_length):
        logits = model(input_ids, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :] / temperature

        # Apply filtering
        next_token_logits = top_k_filtering(next_token_logits, top_k)
        next_token_logits = top_p_filtering(next_token_logits, top_p)

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return text


def main():
    args = parse_args()
    set_seed(args.seed)

    config = GPTConfig.from_yaml(args.config)

    logger = setup_json_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        project_name="gpt-lm-tiny",
        config=config.to_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id

    collate_fn_obj = CollateFn(
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
    )

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:50000]

    train_dataset = CLMDataset(texts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_obj,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTForPreTraining(config).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler()

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0

    print(f"Starting training for {args.num_steps} steps...")
    while global_step < args.num_steps:
        for batch in train_loader:
            if global_step >= args.num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["encoder_attention_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(input_ids, attention_mask=attention_mask)
                # Shift for causal LM: predict next token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
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
                print(f"Step {global_step} | LM Loss: {avg_loss:.4f}")
                log_metrics({"train/lm_loss": avg_loss, "step": global_step})
                running_loss = 0.0

            if global_step % args.eval_every == 0:
                val_texts = ds["text"][50000:51000]
                val_dataset = CLMDataset(val_texts)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn_obj,
                )
                val_loss, val_ppl = evaluate(
                    model, val_loader, device, config.vocab_size, config.pad_token_id
                )
                print(f"Step {global_step} | Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
                log_metrics({
                    "val/lm_loss": val_loss,
                    "val/perplexity": val_ppl,
                    "step": global_step,
                })

                # Qualitative generation
                prompt = "Once upon a time"
                generated = generate_text(model, tokenizer, prompt, args.max_gen_length, device)
                print(f"Generated: {generated}")
                log_metrics({"generated_text": generated, "step": global_step})

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()