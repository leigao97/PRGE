import argparse

import torch
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    DataCollatorForTokenClassification,
    BitsAndBytesConfig
)

import wandb

from lora import LoraConfig, get_peft_model, print_trainable_parameters
from data_loader import load_data, encode_data
from trainer import Trainer
import os
import signal



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sst2", 
                        choices=["sst2", "rte", "boolq", "wsc", "wic", "multirc", "copa", "winogrande", "squad", "drop", "mrpc",
                                 "qqp", "qnli", "wnli"
                                 ],
                        help="The name of the GLUE task to train on.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", 
                        choices=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "facebook/opt-1.3b", "facebook/opt-13b" ,"google/gemma-2b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.2-1B"],
                        help="Path to pretrained model.")
    parser.add_argument("--optimizer", type=str, default="zo", choices=["fo", "zo"], help="Optimizer.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input sequence length after tokenization.")
    parser.add_argument("--n", type=int, default=1, help="Number of pertubation.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--eps", type=float, default=1e-2, help="Perturbation scale.")
    parser.add_argument("--num_train_epochs", type=int, default=1000, help="Total number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--quantization", type=str, default="no", choices=["no", "8bit", "4bit"], help="Quantization.")
    parser.add_argument("--zo_mode", type=str, default="single", choices=["single", "dual"], help="Number of forward pass.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--max_iterations", type=int, default=25000, help="Log every X epochs.")
    parser.add_argument("--split",  type=str2bool, default=False , help="Split batch between perturbations.")
    parser.add_argument("--mixed_precision",  type=str2bool, default=False , help="Split batch between perturbations.")
    parser.add_argument('--spsa_config', type=str, default=None,
                        help='JSON string containing n, per_device_train_batch_size, and split')
    parser.add_argument('--peft', type=str, choices=["no", "lora", "lora-fa"], default="lora-fa", help="PEFT method.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--compute_dtype", type=str, default="fp16", choices=["bf16" ,"fp16", "fp32"], help="Compute dtype.")
    parser.add_argument('--total_batch_size', type=int, default=None, help='Total batch size for training')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if args.total_batch_size is not None:
        assert args.total_batch_size == args.per_device_train_batch_size * num_gpus, "Total batch size must be equal to per_device_train_batch_size * num_gpus"

    if args.spsa_config is not None:
        print("Overriding args with SPSA config")
        import json
        spsa_config = json.loads(args.spsa_config)
        args.n = spsa_config["n"]
        args.per_device_train_batch_size = spsa_config["per_device_train_batch_size"]
        args.split = spsa_config["split"]
        print(f"SPSA config: {spsa_config}")

    print(f"Run args: {args}")
    set_seed(args.seed)
    # compute_dtype = torch.float32 if args.optimizer == "fo" else torch.float16
    compute_types = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


    if args.optimizer == "fo":
        args.compute_dtype = "fp32"
    compute_dtype = compute_types[args.compute_dtype]

    run_name = f"test_v2_{args.task_name}-{args.optimizer}-{args.model_name_or_path}-n{args.n}-bs{args.per_device_train_batch_size}-lr{args.learning_rate}-eps{args.eps}-epochs{args.num_train_epochs}_seed{args.seed}_split{args.split}_mp{args.mixed_precision}_dtype{compute_dtype}_peft{args.peft}_r{args.lora_rank}_alpha{args.lora_alpha}"
    wandb.init(project="MeZO-LLM", name=run_name, config=args)
    print(f"Run name: {run_name}")
        

    if args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=compute_dtype, quantization_config=quantization_config, device_map="auto")
    print(f"Memory footprint (GB): {model.get_memory_footprint() / 1024**3}") 

    if args.peft != "no":
        lora_config = LoraConfig(
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            n=args.n,
            # target_modules=["q_proj", "v_proj"] if "Tiny" not in args.model_name_or_path else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules=["q_proj", "v_proj"],
            zo_mode=args.zo_mode)

        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    if args.peft in ["lora", "lora-fa"]:
        for name, param in model.named_parameters():
            if (args.peft == "lora-fa" and "lora_B" in name) or (args.peft == "lora" and "lora" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.peft == "no":
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown peft mode: {args.peft}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side="left", truncation_side="left")
    if "Llama" in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        print("Llama model tokenizer: set pad token")

    raw_datasets = load_data(args)
    train_dataset, eval_dataset, cls_idx = encode_data(args, tokenizer, raw_datasets)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    optimizer = None

    if args.optimizer == "fo":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        accelerator = Accelerator(mixed_precision=('bf16' if args.mixed_precision else 'no'))
        print(f"Using mixed precision: {args.mixed_precision} with compute dtype: bf16")
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model,
            optimizer,
            DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,  drop_last=args.split),
            DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        )
    else:
        accelerator = Accelerator()
        model, train_dataloader, eval_dataloader = accelerator.prepare(
            model,
            DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,  drop_last=args.split),
            DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        )

    trainer = Trainer(args, model, tokenizer, train_dataloader, eval_dataloader, accelerator, cls_idx, optimizer)
    trainer.train()

    model.save_pretrained("output")
    tokenizer.save_pretrained("output")

    wandb.finish()


if __name__ == "__main__":
    main()
