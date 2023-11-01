from datasets import load_dataset, load_metric

from datasets import ClassLabel
import random
import pandas as pd
import os

import json

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
# from transformers import EarlyStoppingCallback

import soundfile as sf

import numpy as np

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from jiwer import wer

from datetime import datetime

class w2v2_predictor:
    # VOCAB_SIZE = 51
    print("come in")
    def _create_timit(self):
        # self.timit = load_dataset("timit_asr")
        self.timit = load_dataset("/home/ying/Thesispackage_Torgo/torgo", data_dir = "/mnt/data/ying/TORGO")
        # self.timit = self.timit.remove_columns(["word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    def _create_dicts(self):
        def _read_dict(self):
            with open('vocab.json') as vocab_file:
                self.unicode_to_numeric_dict = json.loads(vocab_file.read())
            with open('str_unic.json') as str_unic_file:
                self.str_to_unicode_dict = json.loads(str_unic_file.read())

            self.unicode_to_str_dict = {value: key for (key, value) in self.str_to_unicode_dict.items()}

        def _generate_dict(self):
            def _extract_all_phones(batch):
                # This line is the phones of the utterance
                for detailed_utterence in batch["phonetic_detail"]:
                    for phone in detailed_utterence['utterance']:
                        all_phones.append(phone)
                vocab = list(set(all_phones))
                return {"vocab": [vocab], "all_phones": [all_phones]}


            ## CREATE DICTIONARY FROM PHONES FOR ENCODING
            all_phones = []

            if self.timit == None:
                self._create_timit()

            vocabs = self.timit.map(_extract_all_phones, batched=True, batch_size=-1, keep_in_memory=True,
                                    remove_columns=self.timit.column_names["train"])

            vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

            vocab_dict = {v: k for k, v in enumerate(vocab_list)}

            ## CONVERT TO UNICODE AND CREATE CORRESPONDING DICT
            # make a copy
            self.unicode_dict = vocab_dict.copy()
            # reverse the dict
            self.unicode_dict = {value: key for (key, value) in self.unicode_dict.items()}
            # make it str to unicode dict
            self.unicode_to_numeric_dict = {key: chr(0x0001F970 + key) for (key, value) in
                                            self.unicode_dict.items()}
            self.unicode_to_numeric_dict = {value: key for (key, value) in self.unicode_to_numeric_dict.items()}
            # unicode to numeric dict
            self.str_to_unicode_dict = {chr(0x0001F970 + key): value for (key, value) in self.unicode_dict.items()}
            self.str_to_unicode_dict = {value: key for (key, value) in self.str_to_unicode_dict.items()}

            ## ADD UNK AND PAD
            self.unicode_to_numeric_dict["[UNK]"] = len(self.unicode_to_numeric_dict)
            self.unicode_to_numeric_dict["[PAD]"] = len(self.unicode_to_numeric_dict)
            self.unicode_to_numeric_dict["|"] = len(self.unicode_to_numeric_dict)
            print(len(self.unicode_to_numeric_dict))

            ## CORRECT FOR UNK AND PAD IN STR DICT
            self.str_to_unicode_dict["[UNK]"] = "[UNK]"
            self.str_to_unicode_dict["[PAD]"] = "[PAD]"
            self.str_to_unicode_dict["|"] = "|"

            self.unicode_to_str_dict = {value: key for (key, value) in self.str_to_unicode_dict.items()}

            ## SAVE DICT TO FILE
            # save unic-numeric (for decoding logits)
            with open('vocab.json', 'w') as vocab_file:
                json.dump(self.unicode_to_numeric_dict, vocab_file)
            # save str-unicode/arpabet_phoneme-unicode (decode back to arphabet)
            with open('str_unic.json', 'w') as string_unic_file:
                json.dump(self.str_to_unicode_dict, string_unic_file)

        try:

            if not os.path.isfile('./vocab.json'):
                #Process timit and create a new dictionary (you will have to re-finetune a model)
                _generate_dict(self)
            else:
                #or just read a previously generated dictionary
                _read_dict(self)
        except Exception as e:
            print("Generating Dicts failed. " + str(e))

    ## GET TIMIT DATASET IN THE FORM READY FOR DICTIONARY CREATION AND ANALYSIS
    def _preprocess_timit(self):
        # for use with timit.map - add a unicode representation of each phone to the dataset
        def to_unicode_fn(batch):
            aux_lst = []
            for detailed_utterance in batch['phonetic_detail']:
                lst = []
                for phone in detailed_utterance['utterance']:
                    lst.append(self.str_to_unicode_dict[phone])
                detailed_utterance['unic_utterance'] = lst[:]
                aux_lst.append(detailed_utterance)
            batch['phonetic_detail'] = aux_lst[:]
            return batch

        ## CONVERT LIST OF PHONES TO STRING OF PHONES
        def delim_phones_fn(batch):
            for detailed_utterance in batch['phonetic_detail']:
                # detailed_utterance['string_utterance'] = '|'.join(detailed_utterance['unic_utterance'])
                detailed_utterance['string_utterance'] = ''.join(detailed_utterance['unic_utterance'])
            return batch

        ## CONVERSION TO 1D ARR
        # ['speech', 'sampling_rate', 'target_phones']
        def speech_file_to_array_fn(batch):
            speech_array, sampling_rate = sf.read(batch["file"])
            batch["speech"] = speech_array
            batch["sampling_rate"] = sampling_rate
            batch["target_phones"] = batch['phonetic_detail']['string_utterance']
            return batch
        try:
            if self.str_to_unicode_dict == None:
                self._create_dicts()

            if self.timit == None:
                self._create_timit()

            self.timit = self.timit.map(to_unicode_fn, batch_size=-1, keep_in_memory=True, batched=True)

            self.timit = self.timit.map(delim_phones_fn, batch_size=-1, keep_in_memory=True, batched=True)

            self.timit = self.timit.map(speech_file_to_array_fn, remove_columns=self.timit.column_names["train"],
                                        num_proc=8)

            ## VALIDATE SHAPE
            rand_int = random.randint(0, len(self.timit["train"]))
            print("Preprocessed Shape Inspection"+"\n"+"*"*26)
            print("Index No:", rand_int)
            print("Target phones:", self.timit["train"][rand_int]["target_phones"])
            print("Input array shape:", np.asarray(self.timit["train"][rand_int]["speech"]).shape)
        except Exception as e:
            print("Preprocessing failed " + str(e))

    ## GENERATE THE PROCESSORS. IF SAVE DETECTED, GENERATE FROM THERE, ELSE CONSTRUCT FROM PREPROCESSED TIMIT.
    def _generate_processors(self):
        ## BUILD PROCESSOR
        if (os.path.isdir('/mnt/data/ying/checkpoints/Torgo/saves_1b') and os.path.isfile('./vocab.json')):
            ## TOKENIZER CLASS
            self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file="/mnt/data/ying/checkpoints/Torgo/saves_1b/vocab.json",
                                                  unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

            ## CREATE FEATURE EXTRACTOR
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                              do_normalize=True, return_attention_mask=False)

            ## CREATE PROCESSOR if self.timit_prepared == None:
            self._preprocess_timit()
            self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

            ## SAVE PROCESSOR
            self.processor.save_pretrained("/mnt/data/ying/checkpoints/Torgo/saves_1b")
        else:
            #PREPROCESS TIMIT AND GENERATE THE MISSING FILES
            self._create_dicts()

            ## REPEAT THE ABOVE
            self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file="/mnt/data/ying/checkpoints/Torgo/saves_1b/vocab.json",
                                                  unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                              do_normalize=True, return_attention_mask=False)

            self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

            self.processor.save_pretrained("/mnt/data/ying/checkpoints/Torgo/saves_1b")


    ## GET TIMIT INTO ITS FINAL FORM BEFORE EVALUATING/TESTING. REQUIRES PROCESSORS AND PREPROCESSED TIMIT
    def process_timit(self):
        ## PROCESS
        def prepare_dataset(batch):
            # check that all files have the correct sampling rate
            assert (
                    len(set(batch["sampling_rate"])) == 1
            ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

            # get the audio data
            batch["input_values"] = self.processor(batch["speech"],
                                                   sampling_rate=batch["sampling_rate"][0]).input_values

            # assign the labels
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["target_phones"], is_split_into_words=False).input_ids

            return batch

        if self.processor == None:
            self._generate_processors()

        self._preprocess_timit()

        self.timit_prepared = self.timit.map(prepare_dataset, remove_columns=self.timit.column_names["train"],
                                             batch_size=8,
                                             batched=True)

    def __init__(self):
        ## DEFINE CLASS VARS
        # Dicts
        self.unicode_to_numeric_dict = None
        self.str_to_unicode_dict = None
        self.unicode_to_str_dict = None
        self.unicode_dict = None

        # Datasets
        self.timit = None
        self.timit_prepared = None

        # Processors
        self.tokenizer = None
        self.feature_extractor = None
        self.processor = None

        # Model
        self.model = None

        self._generate_processors()

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    def set_model(self, ckpt):
        # config = Wav2Vec2ForCTC.from_pretrained("/content/drive/MyDrive/Thesispackage/wav2vec2-base-torgo-phones/checkpoint-9500")
        # config.vocab_size = VOCAB_SIZE

        if self.processor == None:
            self._generate_processors()

        if ckpt is None:
            # state_dict = torch.load(f"/home/ying/Thesispackage_Torgo/wav2vec2-large-xlsr-53-torgoTD-checkpoints/4bs25e/checkpoint-10500/pytorch_model.bin", map_location='gpu')
            # state_dict.pop('lm_head.weight')
            # state_dict.pop('lm_head.bias')
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xls-r-1b",
                # state_dict=state_dict,
                # ignore_mismatched_sizes=True,
                gradient_checkpointing=True,
                ctc_loss_reduction="mean",
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=len(self.processor.tokenizer),
            )
        else:
            # state_dict = torch.load(f"/home/ying/Thesispackage_Torgo/wav2vec2-large-xlsr-53-torgoTD-checkpoints/4bs25e/checkpoint-10500/pytorch_model.bin", map_location='gpu')
            # state_dict.pop('lm_head.weight')
            # state_dict.pop('lm_head.bias')
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "/mnt/data/ying/checkpoints/Torgo/wav2vec2_1b_TORGO/8bs50e_1b_1e-5/checkpoint-4250",
                # config = self.config,
                # ignore_mismatched_sizes=True,
                gradient_checkpointing=True,
                ctc_loss_reduction="mean",
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=len(self.processor.tokenizer),
            )

    def train(self, ckpt = None):
        def character_error_rate(pred_str, label_str):
            preds = [char for seq in pred_str for char in list(seq)]
            refs = [char for seq in label_str for char in list(seq)]
            cer = wer(refs, preds)
            return cer

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.batch_decode(pred_ids)
            
            # we do not want to group tokens when computing the metrics
            label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

            error = character_error_rate( pred_str, label_str)
            print(pred_str)
            print(error)
            return {"cer": error}
        try:
            if self.processor == None:
                self._generate_processors()
            self.data_collator = self.DataCollatorCTCWithPadding(processor=self.processor, padding=True)

            self.set_model(ckpt)
            self.model.freeze_feature_extractor()

            training_args = TrainingArguments(
                # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
                output_dir="/mnt/data/ying/checkpoints/Thesispackage_Torgo/wav2vec2-large-xlsr-53-torgoTD-checkpoints/4bs10e-PHN-SN-divided-same-as-SSD-group",
                group_by_length=True,
                gradient_checkpointing=True,
                per_device_train_batch_size=32,
                evaluation_strategy="steps",
                num_train_epochs=50,
                fp16=True,
                save_steps=250,
                eval_steps=250,
                logging_steps=500,
                learning_rate=1e-4,
                weight_decay=0.005,
                warmup_steps=1000,
                save_total_limit=2,
                report_to="wandb",  # enable logging to W&B
                run_name=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')  # name of the W&B run (optional)
            )

            trainer = Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=self.timit_prepared["train"],
                eval_dataset=self.timit_prepared["test"],
                tokenizer=self.processor.feature_extractor,
                # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
            )

            # trainer.train()

        except AssertionError as A:
            print("Model not initialised. Please call .set_model() first.")



    def eval(self, ckpt = None, proc = None):
        def character_error_rate(pred_str, label_str):
            preds = [char for seq in pred_str for char in list(seq)]
            refs = [char for seq in label_str for char in list(seq)]
            cer = wer(refs, preds)
            return cer

        def map_to_result(batch):
            model.to("cuda")
            input_values = self.processor(
                batch["speech"],
                sampling_rate=batch["sampling_rate"],
                return_tensors="pt"
            ).input_values.to("cuda")

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids)[0]

            return batch

        def average_character_error_rate(pred_list, label_list):
            errList = list()
            assert len(pred_list) == len(label_list), "Prediction list and label list must be of equal length"

            for index in range(len(pred_list)):
                errList.append(character_error_rate(pred_list[index], label_list[index]))

            total = sum(errList)
            avg = total / len(pred_list)

            return avg

        try:
            if ckpt != None:
                model = Wav2Vec2ForCTC.from_pretrained(ckpt)
            elif self.model != None:
                model = self.model
            else:
                raise AssertionError("Model not set in ckpt or previously")

            if proc != None:
                processor = Wav2Vec2Processor.from_pretrained(proc)
            elif self.processor != None:
                processor = self.processor
            else:
                self._generate_processors()

            results = self.timit["test"].map(map_to_result)

            print("Test WER: {:.3f}".format(
                average_character_error_rate(pred_list=results["pred_str"], label_list=results["target_phones"])))

        except AssertionError:
            print("Assertion Error. Perhaps you did not set the model or processors?")

    def pred_wav_no_collapse(self, wavpath, ckpt=None, return_type = "phones"):
        try:
            if ckpt != None:
                #print("Predcting using given checkpoint at "+ckpt+"\n Note: Stored model not updated with ckpt")
                model = Wav2Vec2ForCTC.from_pretrained(ckpt)

            elif self.model != None:
                #print("Predicting using stored model")
                model = self.model

            else:
                #print("Predicting using non-finetuned model")
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b")

            model.to("cuda")

            # load wav
            wav_arr = sf.read(wavpath)

            input_values = self.processor(  wav_arr[0],
                                            sampling_rate=wav_arr[1],
                                            return_tensors="pt"
                                            ).input_values.to("cuda")

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            # print(pred_ids)
            # Different part: convert ids to tokens
            # pred_ids: tensor(numericID)
            # tokens: unicode
            tokens = self.processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())


            if return_type == "tokens":
                return tokens
            elif return_type == "phones":
                # phones: arpabet phonemes
                phones = ["".join(self.unicode_to_str_dict[token]) for token in tokens]
                return phones
            else:
                raise AttributeError("Return Type must be 'tokens' or 'phones'")
        except Exception as e:
            print("Predicting Failed. " + str(e))

    def pred_wav_with_collapse(self, wavpath, ckpt=None, return_type="phones"):
        try:
            if ckpt != None:
                #print("Predcting using given checkpoint at " + ckpt + "\n Note: Stored model not updated with ckpt")
                model = Wav2Vec2ForCTC.from_pretrained(ckpt)

            elif self.model != None:
                #print("Predicting using stored model")
                model = self.model

            else:
                #print("Predicting using non-finetuned model")
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b")

            model.to("cuda")

            # load wav
            wav_arr = sf.read(wavpath)

            input_values = self.processor(wav_arr[0],
                                          sampling_rate=wav_arr[1],
                                          return_tensors="pt"
                                          ).input_values.to("cuda")

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)

            # Different part: convert ids to tokens
            tokens = self.processor.batch_decode(pred_ids)[0]

            if return_type == "tokens":
                return tokens
            elif return_type == "phones":
                phones = ["".join(self.unicode_to_str_dict[token]) for token in tokens]
                return phones
            else:
                raise AttributeError("Return Type must be 'tokens' or 'phones'")
        except Exception as e:
            print("Predicting Failed. " + str(e))

