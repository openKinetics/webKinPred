---
language: en
license: mit
tags:
  - bert
  - masked-language-modeling
  - molecular-representation
datasets:
  - chembl
  - zinc15
metrics:
  - roc-auc
  - mae
  - rmse
---

# Example usage
```python
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

# Load the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained('thaonguyen217/farm_molecular_representation')
model = BertForMaskedLM.from_pretrained('thaonguyen217/farm_molecular_representation')

# Example usage
input_text = "N_primary_amine N_secondary_amine c_6-6 1 n_6-6 n_6-6 c_6-6 c_6-6 2 c_6-6 c_6-6 c_6-6 c_6-6 c_6-6 1 2" # FG-enhanced representation of NNc1nncc2ccccc12
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model(**inputs, output_hidden_states=True)

# Extract atom embeddings from last hidden states
last_hidden_states = outputs.hidden_states[-1][0] # last_hidden_states: (N, 768) with N is input length
```
*Note:* For more information about generating FG-enhanced SMILES, please visit this [GitHub repository](https://github.com/thaonguyen217/farm_molecular_representation).

## Purpose

This model aims to:
- Enhance molecular representation by directly incorporating functional group information directly into the representations.
- Facilitate tasks such as molecular prediction, classification, and generation.

# Farm Molecular Representation Model
You can read more about the model in our [paper](https://arxiv.org/pdf/2410.02082) or [webpage](https://thaonguyen217.github.io/farm/) or [github repo](https://github.com/thaonguyen217/farm_molecular_representation).

![FARM](./main.png)
*(a) FARM’s molecular representation learning model architecture. (b) Functional group-aware tokenization and fragmentation algorithm. (c) Snapshot of the functional group knowledge graph. (d) Generation of negative samples for contrastive learning.*
## Overview

The **FARM** (Molecular Representation Model) is designed for molecular representation tasks using a BERT-based approach. The key innovation of FARM lies in its functional group-aware tokenization, which incorporates functional group information directly into the representations. This strategic reduction in tokenization granularity, intentionally interfaced with key drivers of functional properties (i.e., functional groups), enhances the model's understanding of chemical language, expands the chemical lexicon, bridges the gap between SMILES and natural language, and ultimately advances the model's capacity to predict molecular properties. FARM also represents molecules from two perspectives: by using masked language modeling to capture atom-level features and by employing graph neural networks to encode the whole molecule topology. By leveraging contrastive learning, FARM aligns these two views of representations into a unified molecular embedding.

## Components

The model includes the following key files:

- **`model.safetensors`**: The main model weights.
- **`config.json`**: Contains configuration parameters for the model architecture.
- **`generation_config.json`**: Configuration for text generation settings.
- **`special_tokens_map.json`**: Mapping of special tokens used by the tokenizer.
- **`tokenizer.json`**: Tokenizer configuration file.
- **`tokenizer_config.json`**: Additional settings for the tokenizer.
- **`.gitattributes`**: Git attributes file specifying LFS for large files.

## Installation

To use the model, you need to install the required libraries. You can do this using pip:

```bash
pip install transformers torch