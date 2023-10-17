
# Enhancing German-to-English Machine Translation using Pre-trained Encoder and Decoder


## Reference
This project is adapted from homework 2(prefixtune) of Anoop Sarkar. Below is the link to it
git clone https://github.com/anoopsarkar/nlp-class-hw.git


##Approach

This project utilizes encoder-decoder transformers for the machine translation task. Specifically, we will employ the Huggingface's Encoder-Decoder Models library and initialize the model with a pre-trained encoder (German distilled BERT base) and a pre-trained decoder(English distilled GPT2). The library will initialize cross attention between the encoder and the decoder with random weights. Subsequently, fine-tuning will be performed on the model using the German-to-English dataset.

##dataset
The dataset used for this project was IWSLT 2014 German-to-English dataset (https://huggingface.co/datasets/bbaaaa/iwslt14-de-en). Hugging Face library was used to load the dataset.

## Installation

Make sure you setup your virtual environment:

    python3.10 -m venv venv
    source venv/bin/activate
    pip install -U -r requirements.txt


## Create output.zip

To create the `output.zip` file for upload to Coursys do:

    python3 zipout.py

For more options:

    python3 zipout.py -h

## Create source.zip

To create the `source.zip` file for upload to Coursys do:

    python3 zipsrc.py

For more options:

    python3 zipsrc.py -h

## Check your accuracy

To check your accuracy on the dev set:

    python3 check.py

For more options:

    python3 check.py -h

In particular use the log file to check your output evaluation:

    python3 check.py -l log

