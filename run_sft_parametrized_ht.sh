# env HF_TOKEN must be set
model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"

for epochs in {2..3}
do
    for lr in "2e-4" "3e-4"
    do
        python sft.py --hf_token $HF_TOKEN --model_id $model_id --epochs $epochs --lr $lr
    done
done