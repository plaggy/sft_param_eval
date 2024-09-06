import os
import torch
import json
import argparse
from datetime import datetime
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import PartialState
from accelerate.utils import gather_object


def evaluate_pred(pred, label, chkpt):
    last_ans_idx = label.split("\\boxed{")[-1].rfind("}")
    answer = label.split("\\boxed{")[-1][:last_ans_idx]
    answer = answer.replace(" ", "")

    pred_chunk_w_ans = None
    if "Answer: {" in pred:
        pred_chunk_w_ans = pred.split("Answer: {")[-1]
    elif "\\boxed{" in pred:
        pred_chunk_w_ans = pred.split("\\boxed{")[-1]
    elif "{" in pred:
        pred_chunk_w_ans = pred.split("{")[-1]
    if pred_chunk_w_ans is None:
        with open("text_matching.txt", "a") as f:
            f.write(f"{chkpt}\t{answer}\t{pred_chunk_w_ans}")
        return 0
    pred_ans_last_idx = pred_chunk_w_ans.rfind("}")
    if pred_ans_last_idx <= 0:
        with open("text_matching.txt", "a") as f:
            f.write(f"{chkpt}\t{answer}\t{pred_chunk_w_ans}")
        return 0
    pred_ans = pred_chunk_w_ans[:pred_ans_last_idx]
    pred_ans = pred_ans.replace(" ", "")

    with open("text_matching.txt", "a") as f:
        f.write(f"{chkpt}\t{answer}\t{pred_ans}")

    return int(pred_ans == answer)


def format_prompt(tokenizer, problem):
    messages = [{
        "role": "user",
        "content": f"<QUESTION>{problem}"
    }]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = tokenizer.bos_token + messages[0]["content"]

    return prompt


def main(args):
    dataset_name = "lighteval/MATH"
    eval_dataset = load_dataset(dataset_name, split="test[:150]")
    problems = eval_dataset["problem"]
    solutions = eval_dataset["solution"]
    indices = list(range(len(problems)))
    if os.path.isfile(args.checkpoints_dir + "/adapter_config.json"):
        checkpoints = [args.checkpoints_dir]
    else:
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
        chkpt_dir = chkpt_dir.split("/")[-1]
        if args.merge_lora:
            model = model.merge_and_unload()
            chkpt_dir += "_merged"

        with distributed_state.split_between_processes(indices) as chunk:
            results = []
            for i in chunk:
                prompt = format_prompt(tokenizer, problems[i])
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to("cuda")

                output = model.generate(**inputs, max_new_tokens=600, temperature=0.0, do_sample=False)
                output = output[0][len(inputs["input_ids"][0]):]
                output = tokenizer.decode(output, skip_special_tokens=True)

                results.append(evaluate_pred(output, solutions[i], chkpt_dir))

            del model
            torch.cuda.empty_cache()

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
    "--checkpoints_dir",
    required=True
)
parser.add_argument(
    "--output_file",
    default="out.jsonl"
)
parser.add_argument(
    "--merge_lora",
    action="store_true"
)

if __name__ == '__main__':
    args = parser.parse_args()
    start = datetime.now()
    main(args)
    print(f"RUNTIME: {(datetime.now() - start) / 60} min")