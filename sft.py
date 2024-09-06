"""
awq/fp16 base lora
awq base full?
merged/unmerged 4bit bnb lora
merged/unmerged awq lora
masked/unmasked instruction
with/without packing
DPO vs SFT
"""
import os
os.environ["HF_HOME"] = "/tmp/.cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, set_seed, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from accelerate import PartialState
import argparse
import torch
import os

# device_string = PartialState().process_index


def main(args):
    dataset_name = "lighteval/MATH"
    output_dir = f"{args.model_id.split('/')[-1].lower()}_peft_train_seq_len_{args.max_seq_length}" \
        f"{'_packed' if args.pack else ''}{'_masked_instr' if args.mask_instructions else ''}_epoch_{args.epochs}_lr_{args.lr}_2k"

    dataset = load_dataset(dataset_name, split="train[:2000]", trust_remote_code=True)
    eval_dataset = load_dataset(dataset_name, split="test[:200]", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def formatting_func(example):
        last_idx = example["solution"].split("\\boxed{")[-1].find("}")
        answer = example["solution"].split("\\boxed{")[-1][:last_idx]

        messages = [
            {
                "role": "user",
                "content": f"<QUESTION>{example['problem']}"
            },
            {
                "role": "assistant",
                "content": f"<ANSWER>Explanation:\n{example['solution']}\nAnswer: {{{answer}}}"
            }
        ]
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = tokenizer.bos_token + "\n".join([turn["content"] for turn in messages]) + tokenizer.eos_token

        example[args.dataset_text_field] = text

        return example

    dataset = dataset.map(formatting_func, remove_columns=["problem", "solution"])
    eval_dataset = eval_dataset.map(formatting_func, remove_columns=["problem", "solution"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if args.mask_instructions:
        response_template = "<ANSWER>"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_id,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "token": args.hf_token,
        "device_map": "auto"
        # "device_map": {'': device_string}
    }

    if "AWQ" in args.model_id:
        # awq quantization config from the model repo is used automatically
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        lora_target_modules=["qkv_proj", "o_proj", "down_proj"]
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage="uint8",
        )
        model = AutoModelForCausalLM.from_pretrained(
            **model_kwargs,
            quantization_config=bnb_config,
        )
        lora_target_modules="all-linear"

    model.enable_input_require_grads()
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules
    )

    sft_config = SFTConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        packing=args.pack,
        output_dir=output_dir,
        dataset_text_field=args.dataset_text_field,
        chars_per_token=3.6,
        max_seq_length=args.max_seq_length,
        report_to="tensorboard",
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        gradient_accumulation_steps=2,
        # gradient_checkpointing_kwargs={'use_reentrant':False}
        # gradient_checkpointing=True
    )

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(-1)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        args=sft_config,
        peft_config=peft_config,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        dataset_kwargs={
            "append_concat_token": False,
            "add_special_tokens": False,
        }
    )

    print(f"\nPROCESSED INPUT EXAMPLE:\n{repr(tokenizer.decode(trainer.train_dataset[0]['input_ids']))}\n")
    trainer.train()
    trainer.save_model(output_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_token",
    required=True
)
parser.add_argument(
    "--model_id",
    default="meta-llama/Meta-Llama-3.1-8B"
)
parser.add_argument(
    "--mask_instructions",
    action="store_true"
)
parser.add_argument(
    "--max_seq_length",
    default=1024,
    type=int
)
parser.add_argument(
    "--dataset_text_field",
    default="text"
)
parser.add_argument(
    "--pack",
    action="store_true"
)
parser.add_argument(
    "--epochs",
    default=3,
    type=int
)
parser.add_argument(
    "--lr",
    default=2e-4,
    type=float
)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
