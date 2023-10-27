
# Enhancing German-to-English Machine Translation using Pre-trained Encoder and Decoder


## Reference
This project is adapted from homework 2(prefixtune) of Anoop Sarkar. Below is the link to it

    git clone https://github.com/anoopsarkar/nlp-class-hw.git


## Approach
This project utilizes encoder-decoder transformers for the machine translation task. Specifically, we will employ the Huggingface's Encoder-Decoder Models library and initialize the model with a pre-trained encoder (German distilled BERT base) and a pre-trained decoder(English distilled GPT2). The library will initialize cross attention between the encoder and the decoder with random weights. Subsequently, fine-tuning will be performed on the model using the German-to-English dataset.

## dataset
The dataset used for this project was IWSLT 2014 German-to-English dataset (https://huggingface.co/datasets/bbaaaa/iwslt14-de-en). Hugging Face library was used to load the dataset.

## Installation

This project is developed on Linux

Make sure you setup your virtual environment:

    python3.10 -m venv venv
    source venv/bin/activate
    pip install -U -r requirements.txt

## Train

The final model is too big to be uploaded. To train a model

    python3 answer/prefixtune.py -f

About 7 epoches reach the optimum 


## Inference on the training, validation and testing dataset

To inference on the training, validation and testing dataset

    python3 zipout.py

For more options:

    python3 zipout.py -h


## Check your accuracy on the dataset

To check your accuracy on the dataset:

    python3 check.py

For more options:

    python3 check.py -h

In particular use the log file to check your output evaluation:

    python3 check.py -l log

## Inference and validate on custom dataset

To inference on the custome dataset:

    python3 answer/prefixtune.py -i data/input/custom.txt > output.txt


To check your accuracy on the custom dataset (I assume you provide custom.out for comparsion):

    python3 bleu.py -t data/reference/custom.out -o output.txt

