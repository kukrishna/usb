## USB: A Unified Summarization Benchmark Across Tasks and Domains

This repository contains the dataset and code for creating the [USB Benchmark](https://arxiv.org/abs/2305.14296). ( Update: The benchmark datasets are now also available on Huggingface at this [link](https://huggingface.co/datasets/kundank/usb)! )

This repository also provides scripts to run and evaluate some models on the benchmark. The models include ChatGPT (via few-shot prompting), and Flan-T5-XL (finetuned on the training set of the benchmark, available at the links listed [here](https://huggingface.co/datasets/kundank/usb)).


Below we provide step-by-step instructions to create the benchmark datasets and run evaluation of the two models:

### Step1
Clone the repository
```shell
git clone https://github.com/kukrishna/usb.git
```


### Step2
Extract the raw document annotations by running the command
```shell
tar -xf raw_annotations.tar.gz
```

### Step3
Install the required libraries (very few) by running 
```shell
pip install -r requirements.txt
```

### Step4
Run the single master script which will create the labeled datasets for all tasks and domains.
The labled datasets will be written in a folder named `task_datasets`.
```shell
bash create_all_datasets.sh
```

### Step5
To ensure that the labeled datasets were created properly, you can match the checksums for the generated
files which we have included in the `checksums.txt` file.

### Step6
We provide pipelines to evaluate models such as ChatGPT or FlanT5 on all tasks.
When you run a pipeline script, it :
(1) converts examples from the task datasets into seq2seq format with task instructions and optional fewshot examples in the input
(2) runs inference using the corresponding model
(3) runs the evaluation script to produce metrics for the model outputs for each task

```shell
cd experiments

# to run chatgpt pipeline
bash run_pipeline_chatgpt.sh

# to run finetuned flant5-xl pipeline
bash run_pipeline_flant5.sh
```


---

More details can be found in the paper:  [https://aclanthology.org/2023.findings-emnlp.592/](https://aclanthology.org/2023.findings-emnlp.592/)


If you use this dataset, please cite it as below:
```

@inproceedings{krishna-etal-2023-usb,
    title = "{USB}: A Unified Summarization Benchmark Across Tasks and Domains",
    author = "Krishna, Kundan  and
      Gupta, Prakhar  and
      Ramprasad, Sanjana  and
      Wallace, Byron  and
      Bigham, Jeffrey  and
      Lipton, Zachary",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
    pages = "8826--8845"
}

```

