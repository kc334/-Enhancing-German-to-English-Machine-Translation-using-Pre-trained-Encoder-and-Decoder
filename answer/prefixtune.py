import argparse, os, string, sys
import torch
from tqdm import tqdm
from transformers import default_data_collator, get_linear_schedule_with_warmup,DistilBertTokenizer,GPT2Tokenizer,EncoderDecoderModel
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TableToText:

    def __init__(
            self,
            modelfile,
            modelsuffix='.pt',
            baseencodermodel='distilbert-base-german-cased',
            basedecodermodel='distilgpt2',
            traindata='bbaaaa/iwslt14-de-en',
            epochs=5,
            batchsize=4,
            lr=5e-5
        ):
        # the input sentences will be handled using this object, you do not need to manually encode input sentence words
        self.tokenizerbert = DistilBertTokenizer.from_pretrained(baseencodermodel)
        self.tokenizergpt2 = GPT2Tokenizer.from_pretrained(basedecodermodel)
        self.tokenizergpt2.pad_token_id = self.tokenizergpt2.eos_token_id
        self.traindata = traindata
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.baseencodermodel = baseencodermodel
        self.basedecodermodel = basedecodermodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.training_data = []
        self.prompt = "Übersetze Deutsch ins Englische: "
        self.model = None # setup the model in self.decode() or self.train()

    def preprocess_function(self, examples):
        source_lang = "de"
        target_lang = "en"
        prefix = self.prompt
        max_length = 128
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        input = self.tokenizerbert(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = self.tokenizergpt2(targets, max_length=max_length, truncation=True, padding="max_length")
        model_inputs ={}
        model_inputs['input_ids']=input['input_ids']
        model_inputs['attention_mask'] = input['attention_mask']
        #cross attention to CLS only
        #model_inputs['attention_mask'] = [ [1]+[0]*(len(sublist)-1) for sublist in input['attention_mask']]
        model_inputs['labels'] = labels["input_ids"]
        return model_inputs

    def get_data(self, splits=("train", "test")):
        """
        Loads the requested dataset with name == :param dataset_name: and returns dataloaders over each split defined
          in :param splits: which can contain any subset of ("train", "validation", "test"). The dataloder batchsize will be
            defined using :param self.batchsize:.
        """
        dataset = load_dataset(self.traindata)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        data_loaders = {}
        for split in splits:
            assert split in processed_datasets
            data_loaders[split] = DataLoader(
                                    processed_datasets[split],
                                    collate_fn=default_data_collator,
                                    batch_size=self.batchsize,
                                    pin_memory=True,
                                    shuffle=(split == "train")
                                  )
        return data_loaders

    def train(self):
        data_loaders = self.get_data(splits=("train", ))
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.baseencodermodel,self.basedecodermodel)

        # You can print the parameters for debugging or understanding the code
        # but make sure you comment it out otherwise it will pollute the output
        # that is produced for dev and test
        #model.print_trainable_parameters()

        # TODO
        # if using HF peft module, then add calls to PrefixTuningConfig and get_peft_model
        # which will take num_virtual_tokens which is set to self.virtualtokens and
        # prefix_projection which is set to self.prefixprojection
        model.config.decoder_start_token_id = self.tokenizergpt2.bos_token_id
        model.config.eos_token_id = self.tokenizergpt2.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(data_loaders["train"]) * self.epochs),
        )

        model = model.to(device)
        for epoch in range(self.epochs):
            model.train()

            # TODO rest of the training steps for prefix tuning
            total_loss = 0
            for step, batch in enumerate(tqdm(data_loaders['train'])):
                assert list(batch.keys()) == ['input_ids', 'attention_mask', 'labels']
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            train_epoch_loss = total_loss / len(data_loaders['train'])
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
            if epoch == self.epochs - 1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            model.save_pretrained(savefile)

    def decode(self, model, inputfile):
        inputpath = Path(inputfile)
        assert inputpath.exists()
        with inputpath.open() as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0 and not line.isspace()]
            decoder_output = []
            for i, src in tqdm(enumerate(lines)):
                predicted_line = self.predict(model, src, num_sequences=1)
                #if not predicted_line or src.split()[0] not in predicted_line.split():
                    # if output generation failed then use a heuristic to generate some output
                    #predicted_line = src.replace(':', '').replace('|', '').replace('  ', ' ')

                decoder_output.append(f"{i}||{predicted_line}")
        return decoder_output

    def predict(self, model, src, num_sequences=1):
        inputs = self.tokenizerbert(self.prompt + src, return_tensors="pt")
        prediction = None
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            #cross attention to CLS only
            #length = len(inputs['attention_mask'][0]) - 1
            #inputs["attention_mask"][0]=torch.tensor([1]+[0]*length)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                eos_token_id=self.tokenizergpt2.eos_token_id,
                pad_token_id=self.tokenizergpt2.eos_token_id,
                do_sample=True,
                num_beams=5,
                top_p=0.9,
                temperature=1.0,
                num_return_sequences=num_sequences
            )
            '''
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=30,
                eos_token_id=self.tokenizergpt2.eos_token_id,
                pad_token_id=self.tokenizergpt2.eos_token_id,
                do_sample=True,
                num_beams=5,
                top_p=0.9,
                temperature=1,
                num_return_sequences=num_sequences,
                no_repeat_ngram_size=5
            )
            '''
            # TODO you may want to generate more than one sequence and choose the best one!
            text = self.tokenizergpt2.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            #return text.lstrip().replace(self.prompt + src, "").replace("\n", " ")
            return text.lstrip().replace("\n", " ")
        
    #calculate perplexity
    def compute_perplexity(self, model, data_loader):
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader)):
                assert list(batch.keys()) == ['input_ids', 'attention_mask', 'labels']
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['attention_mask'].sum().item()
                total_tokens += batch['attention_mask'].sum().item()
                #print(f"Batch {step}, Loss: {loss.item()}, Tokens: {batch['attention_mask'].sum().item()}")

        average_loss = total_loss / total_tokens
        perplexity_score = torch.exp(torch.tensor(average_loss)).item()
        return perplexity_score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'dev.txt'),
                             help="produce table to text output for these input tables")
    argparser.add_argument("-t", "--traindata", dest="traindata",
                            default='bbaaaa/iwslt14-de-en',
                            help="name of hugging face cleaned up dataset for the E2E table to text task")

    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'bert2gpt2'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--baseencodermodel", dest="baseencodermodel",
                           default='distilbert-base-german-cased',
                           help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-N", "--basedecodermodel", dest="basedecodermodel",
                            default='distilgpt2',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1,
                            help="number of epochs [default: 1]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16,
                            help="batch size [default: 8]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')
    table_to_text = TableToText(
                        modelfile,
                        modelsuffix=opts.modelsuffix,
                        baseencodermodel=opts.baseencodermodel,
                        basedecodermodel=opts.basedecodermodel,
                        traindata=opts.traindata,
                        epochs=opts.epochs,
                        batchsize=opts.batchsize,
                        lr=opts.lr
                    )

    # TODO default.py always uses a prompt to produce output from the pretrained model
    # when you have implemented prefix tuning then change this to False to train and/or
    # use your prefix tuned model
    model = None
    if False:
        print(f"Loading the non-finetuned pre-trained model: {opts.baseencodermodel,opts.basedecodermodel}", file=sys.stderr)
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(opts.baseencodermodel,opts.basedecodermodel)
        tokenizergpt2 = GPT2Tokenizer.from_pretrained(opts.basedecodermodel)
        model.config.decoder_start_token_id = tokenizergpt2.bos_token_id
        model.config.eos_token_id = tokenizergpt2.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model = model.to(device)
    else:
        if not os.path.isdir(modelfile + opts.modelsuffix) or opts.force:
            print(f"Could not find modelfile {modelfile + opts.modelsuffix} or -f used. Starting training.", file=sys.stderr)
            table_to_text.train()
            print("Training done.", file=sys.stderr)
        # use the model file if available and opts.force is False
        assert(os.path.isdir(modelfile + opts.modelsuffix))
        print(f"Found modelfile {modelfile + opts.modelsuffix}. Starting decoding.", file=sys.stderr)
        # TODO: if using hf peft library for prefix tuning:
        model = EncoderDecoderModel.from_pretrained(modelfile + opts.modelsuffix)
        model = model.to(device)
    if model:
        decoder_output = table_to_text.decode(model, opts.inputfile)
        print("\n".join(decoder_output))

        # probably dont want this one 
        validation_data_loader = table_to_text.get_data(splits=("validation", ))["validation"]
        perplexity_score_validation = table_to_text.compute_perplexity(model, validation_data_loader)
        print(f"Perplexity on validation data: {perplexity_score_validation}")

        # Calculate perplexity on test data
        test_data_loader = table_to_text.get_data(splits=("test", ))["test"]
        perplexity_score = table_to_text.compute_perplexity(model, test_data_loader)
        print(f"Perplexity on test data: {perplexity_score}")


