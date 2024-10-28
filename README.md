# **SC-MAMBA2: Leveraging State-Space Models for Efficient Single-Cell Ultra-Long Transcriptome Modeling**

This repository contains the code for the `SC-MAMBA2` package for modeling single-cell transcriptome data.

In [our study](https://www.biorxiv.org/content/10.1101/2024.09.30.615775v1), we applied the state-space model [MAMBA2](https://arxiv.org/pdf/2405.21060) to single-cell transcriptomics, developing the SC-MAMBA2 model. SC-MAMBA2 leverages the efficiency and scalability of state-space models (SSM), enabling it to handle ultra-long transcriptome sequences with lower computational cost.

The model was trained on a dataset of 57 million cells, making it the most comprehensive solution to date for processing ultra-long sequences. Extensive benchmarking across various downstream tasks consistently demonstrated SC-MAMBA2's superior performance compared to state-of-the-art models, showcasing exceptional accuracy and computational efficiency.

## **Method Overview**
![Workflow](https://raw.githubusercontent.com/GlancerZ/scMamba2/main/docs/model_arch.png)

## **Installation**

To install the required dependencies for SC-MAMBA2, run the following command:

```sh
pip install -r requirements.txt
```
Important: Make sure that the versions of `mamba-ssm` and `causal-conv1d` are compatible with your Python and CUDA versions. 

## **Resource Data**

Download the dataset used in the paper from the Data folder.

## **Finetuning SC-MAMBA2**

SC-MAMBA2 supports a variety of finetuning tasks to enhance performance on specific single-cell analysis applications. Below are some of the key finetuning tasks:

### **1. Batch Correction**

SC-MAMBA2 can be finetuned to address batch effects in single-cell datasets. By incorporating domain adaptation techniques and leveraging the state-space modeling capabilities, SC-MAMBA2 effectively harmonizes data collected from multiple experimental batches, ensuring a more consistent downstream analysis.

### **2. Cell-type Annotation**

The model can be finetuned to accurately annotate cell types across diverse single-cell datasets. By training on labeled data, SC-MAMBA2 achieves high accuracy in classifying cell types even within highly heterogeneous populations, making it an essential tool for understanding cell diversity in biological samples.

### **3. Multi-omics Integration**

SC-MAMBA2 can integrate information from different omics layers, such as transcriptomics, proteomics, and epigenomics, to generate a comprehensive view of cellular states. Finetuning SC-MAMBA2 for multi-omics integration allows researchers to derive deeper insights into the interplay between different biological pathways and regulatory networks.

### **4. Perturbation & Reverse Perturbation**

SC-MAMBA2 can be finetuned to model cellular responses to perturbations, such as drug treatments or genetic modifications. The reverse perturbation functionality further allows the model to predict the necessary interventions to achieve desired cellular states, providing a powerful tool for therapeutic development and precision medicine.
