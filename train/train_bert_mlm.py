# train/train_bert_mlm.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.config import BertConfig
from utils.logging import setup_json_logging, log_metrics
from utils.helpers import set_seed
from models.encoder_only import BertForPreTraining

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bert_tiny.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/bert_mlm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=60000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    return parser.parse_args()

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def mlm_mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    labels = input_ids.clone()

    # Identify special tokens
    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(
                seq.tolist(), already_has_special_tokens=True
            )
            for seq in input_ids
        ],
        dtype=torch.bool,
    )

    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% → [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% → random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_tokens = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long
    )
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% → unchanged
    return input_ids, labels


def collate_fn(batch, tokenizer, max_length=128, mlm_probability=0.15):
    encoding = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    input_ids, labels = mlm_mask_tokens(
        input_ids, tokenizer, mlm_probability
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, vocab_size):
    model.eval()
    total_loss = 0.0
    total_masked = 0
    correct = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        total_loss += loss.item() * labels.size(0)

        mask = labels != -100
        if mask.any():
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total_masked if total_masked > 0 else 0.0
    model.train()
    return avg_loss, accuracy

class CollateFn:
    def __init__(self, tokenizer, max_length, mlm_probability):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        return collate_fn(
            batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mlm_probability=self.mlm_probability,
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    config = BertConfig.from_yaml(args.config)

    logger = setup_json_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        project_name="bert-mlm-tiny",
        config=config.to_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id

    collate_fn_obj = CollateFn(
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        mlm_probability=config.mlm_probability,
    )

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:50000]

    train_dataset = MLMDataset(texts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_obj,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForPreTraining(config).to(device)

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

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
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
                print(f"Step {global_step} | MLM Loss: {avg_loss:.4f}")
                log_metrics({"train/mlm_loss": avg_loss, "step": global_step})
                running_loss = 0.0


            if global_step % args.eval_every == 0:
                val_texts = ds["text"][50000:51000]
                val_dataset = MLMDataset(val_texts)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn_obj,
                )

                val_loss, val_acc = evaluate(
                    model, val_loader, device, config.vocab_size
                )

                print(
                    f"Step {global_step} | "
                    f"Val Loss: {val_loss:.4f} | MLM Acc: {val_acc:.4f}"
                )

                log_metrics(
                    {
                        "val/mlm_loss": val_loss,
                        "val/mlm_accuracy": val_acc,
                        "step": global_step,
                    }
                )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        os.path.join(args.output_dir, "pytorch_model.bin"),
    )
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()


