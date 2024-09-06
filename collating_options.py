from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig, set_seed, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

mask_instructions_values = [True, False]
packing_values = [True, False]
max_seq_length_values = [256, 512]
dataset_text_field = "text"
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

data_sample = [
    {
        "problem": "Let \[f(x) = \left\{ \\begin{array}{cl} ax+3, &\\text{ if }x>2, \\ x-5 &\\text{ if } -2 \le x \le 2, \\ 2x-b &\\text{ if } x <-2. \end{array} \\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).",
        "solution": """For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."""
    },
    {
        "problem": "A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?",
        "solution": "Let $x$ be the number of band members in each row for the original formation, when two are left over. Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members. Based on the first equation, we must have $rx$ less than 98. We can guess and check some values of $r$ and $x$ in the last equation. If $r=18$, then $x=7$, and $rx=126$ which is too big. If $r=16$, then $x=6$, and $rx=96$, which is less than 98. Checking back in the second formation, we see that $(16-2)(6+1)=14\cdot 7=98$ as it should. This is the best we can do, so the largest number of members the band could have is $\\boxed{98}$."
    },
    {
        "problem": "What is the degree of the polynomial $(4 +5x^3 +100 +2\pi x^4 + \sqrt{10}x^4 +9)$?",
        "solution": "This polynomial is not written in standard form. However, we don't need to write it in standard form, nor do we need to pay attention to the coefficients. We just look for the exponents on $x$. We have an $x^4$ term and no other term of higher degree, so $\\boxed{4}$ is the degree of the polynomial."
    },
    {
        "problem": "Evaluate $\left\lceil3\left(6-\\frac12\\right)\\right\\rceil$.",
        "solution": "Firstly, $3\left(6-\\frac12\\right)=18-1-\\frac12=17-\\frac12$. Because $0\le\\frac12<1$, we have $\left\lceil17-\\frac12\\right\\rceil=\\boxed{17}$."
    }
]

dataset = Dataset.from_list(data_sample)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def formatting_func(example):
    last_idx = example["solution"].split("\\boxed{")[-1].find("}")
    answer = example["solution"].split("\\boxed{")[-1][:last_idx]

    messages = [
        {
            "role": "user",
            "content": f"Question: {example['problem']}"
        },
        {
            "role": "assistant",
            "content": f"<ANSWER>\nExplanation:\n{example['solution']}\nAnswer: {answer}"
        }
    ]
    example[dataset_text_field] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

dataset = dataset.map(formatting_func)

class FakeTrainer:
    _dataset_sanity_checked = False
    dataset_num_proc = 1
    dataset_batch_size = 4

for mask_instructions in mask_instructions_values:
    for pack in packing_values:
        for max_seq_length in max_seq_length_values:

            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            if mask_instructions:
                response_template = "<ANSWER>\n"
                collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

            if pack:
                processed = SFTTrainer._prepare_packed_dataloader(
                    self=None,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    dataset_text_field=dataset_text_field,
                    max_seq_length=max_seq_length,
                    num_of_sequences=1024, # use default from ConstantLengthDataset
                    chars_per_token=3.6, # default
                    # formatting_func, # skipping - already formatted
                    append_concat_token=False, # `eos_token_id` is already at the end of each sample
                    add_special_tokens=False, # I think it's done already
                )
            else:
                processed = SFTTrainer._prepare_non_packed_dataloader(
                    self=FakeTrainer(),
                    tokenizer=tokenizer,
                    dataset=dataset,
                    dataset_text_field=dataset_text_field,
                    max_seq_length=max_seq_length,
                    # formatting_func, # skipping - already formatted
                    add_special_tokens=False,
                    remove_unused_columns=True,
                )

            # for item in processed:
            #     print(tokenizer.decode(item["input_ids"]))

            processed = processed.to_list()
            collated = collator(processed)

            print("--------------\n")
            print(f"Parameters:\nmask_instructions: {mask_instructions}\npack: {pack}\n"
                  f"max_seq_length: {max_seq_length}\n\n")

            for ids, labels in zip(collated["input_ids"].tolist(), collated["labels"].tolist()):
                print(f"Input:\n{tokenizer.decode(ids)}")
                valid_labels = [l for l in labels if l >= 0]
                print(f"\n\nLabel:\n{tokenizer.decode(valid_labels)}\n\n")
