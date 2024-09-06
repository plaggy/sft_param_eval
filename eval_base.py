import os
import torch
import json
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import PartialState
from accelerate.utils import gather


def evaluate_pred(pred, label):
    last_ans_idx = label.split("\\boxed{")[-1].rfind("}")
    answer = label.split("\\boxed{")[-1][:last_ans_idx]

    pred_chunk_w_ans = None
    if "\\boxed{" in pred:
        pred_chunk_w_ans = pred.split("\\boxed{")[-1]
    elif "Answer: {" in pred:
        pred_chunk_w_ans = pred.split("Answer: {")[-1]
    elif "{" in pred:
        pred_chunk_w_ans = pred.split("{")[-1]
    if pred_chunk_w_ans is None:
        return 0
    pred_ans_last_idx = pred_chunk_w_ans.rfind("}")
    if pred_ans_last_idx <= 0:
        return 0
    pred_ans = pred_chunk_w_ans[:pred_ans_last_idx]
    pred_ans = pred_ans.replace(" ", "")
    answer = answer.replace(" ", "")

    # with open("pred_vs_ans.txt", "a") as f:
    #     f.write(f"{chkpt_dir}\t{answer}\t{pred_ans}\n")

    return int(pred_ans == answer)


def main(args):
    dataset_name = "lighteval/MATH"
    eval_dataset = load_dataset(dataset_name, split="test[:150]")
    problems = eval_dataset["problem"]
    solutions = eval_dataset["solution"]
    indices = list(range(len(problems)))
    checkpoints = ["meta-llama/Meta-Llama-3.1-8B"]

    for chkpt_dir in checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(chkpt_dir, token=args.hf_token)

        # distributed_state = PartialState()

        model_kwargs = {
            "pretrained_model_name_or_path": chkpt_dir,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "token": args.hf_token,
            "device_map": "auto"
        }

        model = AutoModelForCausalLM.from_pretrained(
            **model_kwargs
        )

        # with distributed_state.split_between_processes(indices) as chunk:
        results = []
        for i in indices:
            messages = [
                {
                    "role": "user",
                    "content": f"<QUESTION>{problems[i]}"
                }
            ]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = tokenizer.bos_token + messages[0]["content"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to("cuda")

            output = model.generate(**inputs, max_new_tokens=500)
            output = output[0][len(inputs["input_ids"][0]):]
            # ideally should be tensors all the way up to gather()
            output = tokenizer.decode(output, skip_special_tokens=True)

            results.append(evaluate_pred(output, solutions[i]))

        # distributed_state.wait_for_everyone()
        # results = torch.cat(gather(results))

        # if distributed_state.is_main_process:
        res = sum(results) / len(results)
        with open(args.output_file, "a") as f:
            f.write(json.dumps({"model": chkpt_dir, "res": res}) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_token",
    required=True
)
parser.add_argument(
    "--output_file",
    default="out_base.jsonl"
)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)