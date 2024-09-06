import os
import torch
import json
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import PartialState
from accelerate.utils import gather_object


def evaluate_batched_pred(preds, labels):
    results = []
    for pred, label in zip(preds, labels):
        last_ans_idx = label.split("\\boxed{")[-1].rfind("}")
        answer = label.split("\\boxed{")[-1][:last_ans_idx]

        pred_chunk_w_ans = None
        if "Answer: {" in pred:
            pred_chunk_w_ans = pred.split("Answer: {")[-1]
        elif "{" in pred:
            pred_chunk_w_ans = pred.split("{")[-1]
        if pred_chunk_w_ans is None:
            results.append(0)
            continue
        pred_ans_last_idx = pred_chunk_w_ans.rfind("}")
        if pred_ans_last_idx <= 0:
            results.append(0)
            continue
        pred_ans = pred_chunk_w_ans[:pred_ans_last_idx]
        pred_ans = pred_ans.replace(" ", "")
        answer = answer.replace(" ", "")

        results.append(int(pred_ans == answer))

    return results


def main(args):
    dataset_name = "lighteval/MATH"
    eval_dataset = load_dataset(dataset_name, split="test[:150]")
    problems = eval_dataset["problem"]
    solutions = eval_dataset["solution"]
    checkpoints = [d.path for d in os.scandir(args.checkpoints_dir) if d.is_dir()]

    distributed_state = PartialState()

    for chkpt_dir in checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(chkpt_dir)
        config = PeftConfig.from_pretrained(chkpt_dir)
        model_id = config.base_model_name_or_path

        model_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "token": args.hf_token,
            "device_map": distributed_state.device
        }

        if "AWQ" in model_id:
            # awq quantization config from the model repo is used automatically
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
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

        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, chkpt_dir)

        def format_prompt(problem):
            messages = [{
                "role": "user",
                "content": f"<QUESTION>{problem}"
            }]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = tokenizer.bos_token + messages[0]["content"]

            return prompt

        prompts = [format_prompt(p) for p in problems]
        batched_prompts = [prompts[i: i + args.batch_size] for i in range(0, len(problems), args.batch_size)]
        batched_solutions = [solutions[i: i + args.batch_size] for i in range(0, len(solutions), args.batch_size)]
        batched_prompts = [
            tokenizer(batch, padding=True, pad_to_multiple_of=8, return_tensors="pt", add_special_tokens=False)
            for batch in batched_prompts
        ]
        indices = list(range(len(batched_prompts)))

        with distributed_state.split_between_processes(indices) as chunk:
            results = []
            for i in chunk:
                batch = batched_prompts[i]
                batch.to(distributed_state.device)
                output = model.generate(**batch, max_new_tokens=500)
                output = [seq_out[len(seq_in):] for seq_out, seq_in in zip(output, batch["input_ids"])]
                output = tokenizer.batch_decode(output, skip_special_tokens=True)

                results.extend(evaluate_batched_pred(output, batched_solutions[i]))

        results = gather_object(results)

        res = sum(results) / len(results)
        with open(args.output_file, "a") as f:
            f.write(json.dumps({"model": chkpt_dir, "res": res}) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_token",
    required=True
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4
)
parser.add_argument(
    "--checkpoints_dir",
    required=True
)
parser.add_argument(
    "--output_file",
    default="out.jsonl"
)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)