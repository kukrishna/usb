
ALL_TASKS=(
multisentence_compression
abstractive_summarization
fixing_factuality
unsupported_span_prediction
topicbased_summarization
factuality_classification
extractive_summarization
evidence_extraction
)


python chatgpt/convert_datasets_for_fewshot.py \
    --dataset-folder-root ../task_datasets/all/ \
    --output-folder-root ./chatgpt/converted_data/all \
#    --dummyrun # this flag makes small datasets so that the script can be tested end-to-end quickly

mkdir chatgpt/outputs
mkdir chatgpt/outputs/raw

for TASK in "${ALL_TASKS[@]}"
do
echo ${TASK}

mkdir chatgpt/outputs/raw/${TASK}

python chatgpt/predict.py \
    --input-file chatgpt/converted_data/all/${TASK}/test.jsonl \
    --output-file chatgpt/outputs/raw/${TASK}/test_output.jsonl
done


mkdir chatgpt/outputs/postproc

for TASK in "${ALL_TASKS[@]}"
do
echo ${TASK}
mkdir chatgpt/outputs/postproc/${TASK}
python chatgpt/postprocess.py \
        --input-file chatgpt/outputs/raw/${TASK}/test_output.jsonl \
        --output-file chatgpt/outputs/postproc/${TASK}/test_output.jsonl

done

python evaluate_all.py --prediction-file "chatgpt/outputs/postproc/*/test_output.jsonl"  --output-file all_metrics_chatgpt.json



