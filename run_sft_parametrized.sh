# env HF_TOKEN must be set

for model_id in "meta-llama/Meta-Llama-3.1-8B" "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
do
    for pack in {0..1}
    do
        for mask in {0..1}
        do
            if [ $pack -eq 0 ] && [ $mask -eq 0 ]
            then
                echo "running $model_id"
                python sft.py --hf_token $HF_TOKEN --model_id $model_id
            elif [ $pack -eq 1 ] && [ $mask -eq 0 ]
            then
                echo "running $model_id --pack"
                python sft.py --hf_token $HF_TOKEN --model_id $model_id --pack
            elif [ $pack -eq 0 ] && [ $mask -eq 1 ]
            then
                echo "running $model_id --mask_instructions"
                python sft.py --hf_token $HF_TOKEN --model_id $model_id --mask_instructions
            else
                continue
            fi
        done
    done
done