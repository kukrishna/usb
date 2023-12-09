
python flant5/convert_datasets_for_flant5.py \
            --dataset-folder-root ../task_datasets/all/ \
                --output-folder-root ./flant5/converted_data/all


ALL_CLASSIFICATION_TASKS=(
factuality_classification
extractive_summarization
evidence_extraction
)

ALL_GENERATION_TASKS=(
multisentence_compression
abstractive_summarization
topicbased_summarization
fixing_factuality
)


ALL_SPANPRED_TASKS=(
unsupported_span_prediction
)


GPU_ID=0
DATASET_ROOT=flant5/converted_data/all/
OUTPUT_DIR=flant5/outputs
mkdir ${OUTPUT_DIR}


for TASK in "${ALL_GENERATION_TASKS[@]}"
  do
    echo ${TASK}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python flant5/predict_freeform.py \
            --input-file ${DATASET_ROOT}/${TASK}/test.jsonl \
            --model-path kundank/usb-${TASK}-flant5xl \
            --task ${TASK} --nbeams 4 \
            --output-file ${OUTPUT_DIR}/${TASK}_test_outputs.jsonl
  done

for TASK in "${ALL_CLASSIFICATION_TASKS[@]}"
    do
      echo ${TASK}
      CUDA_VISIBLE_DEVICES=${GPU_ID} python flant5/predict_classes.py \
              --input-file ${DATASET_ROOT}/${TASK}/test.jsonl \
              --model-path kundank/usb-${TASK}-flant5xl \
              --task ${TASK} --nbeams 1 \
            --output-file ${OUTPUT_DIR}/${TASK}_test_outputs.jsonl
    done


for TASK in "${ALL_SPANPRED_TASKS[@]}"
    do
      echo ${TASK}
      CUDA_VISIBLE_DEVICES=${GPU_ID} python flant5/predict_spans.py \
              --input-file ${DATASET_ROOT}/${TASK}/test.jsonl \
              --model-path kundank/usb-${TASK}-flant5xl \
              --task ${TASK} --nbeams 4 \
            --output-file ${OUTPUT_DIR}/${TASK}_test_outputs.jsonl
    done

python evaluate_all.py --prediction-file ${OUTPUT_DIR}/"*_test_outputs.jsonl"  --output-file all_metrics_flant5xl.json



