## USB: A Unified Summarization Benchmark Across Tasks and Domains



This repository contains the dataset and code for creating the [USB Benchmark](https://arxiv.org/abs/2305.14296).
Follow the steps below to start working with it:

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

---

More details can be found in the paper: [https://arxiv.org/abs/2305.14296](https://arxiv.org/abs/2305.14296)

If you use this dataset, please cite it as below:
```
@misc{krishna2023usb,
      title={USB: A Unified Summarization Benchmark Across Tasks and Domains}, 
      author={Kundan Krishna, Prakhar Gupta, Sanjana Ramprasad, Byron C. Wallace, Jeffrey P. Bigham, Zachary C. Lipton},
      year={2023},
      eprint={2305.14296},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

