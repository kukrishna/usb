
DATASET_REPO_ROOT="raw_annotations"
DATASET_OUTPUT_ROOT="task_datasets"

echo "READING DATASET FROM REPOSITIORY HERE: " ${DATASET_REPO_ROOT}
echo "WILL WRITE ALL TASK DATASETS TO : " ${DATASET_OUTPUT_ROOT}


ALL_VARIANTS=(
"biographies"
"companies"
"disasters"
"schools"
"landmarks"
"newspapers"
"all"
)

for VARIANT_SIG in "${ALL_VARIANTS[@]}"
do
#COMP
#ABS
python dataset_creators/usb_abs_comp.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#EVEXT
python dataset_creators/usb_evext.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#FAC
python dataset_creators/usb_fac.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#FIX
python dataset_creators/usb_fix.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#EXT
python dataset_creators/usb_ext.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#TOPIC
python dataset_creators/usb_topic.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

#UNSUP
python dataset_creators/usb_unsup.py --split-path ${DATASET_REPO_ROOT}/splits/${VARIANT_SIG} --output-path ${DATASET_OUTPUT_ROOT}/${VARIANT_SIG} --annot-path ${DATASET_REPO_ROOT}/annotations

done


