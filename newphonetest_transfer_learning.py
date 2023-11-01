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
from transformers import Wav2Vec2Config


import soundfile as sf

import numpy as np

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from jiwer import wer

from datetime import date, datetime
import pytz


def main():
    today_date = date.today()
    today_date_str = today_date.strftime("%d-%m-%Y") 
    set_timezone = pytz.timezone('Australia/Perth')
    today_time = datetime.now(set_timezone).time()
    today_time_str = today_time.strftime("%H:%M:%S") 
    print("Time = ", today_date_str, today_time_str) 
    print("****** in newphonetest ******")
    # timit = load_dataset("/home/ying/Thesispackage_Torgo_SSD/torgo", data_dir = "/mnt/data/ying/TorgoAmendPhnFiles_PHN_SN")
    timit = load_dataset("timit_asr")

    print(timit)
    VOCAB_SIZE = 50

    timit = timit.remove_columns(["word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    # def wrap_phones_fn(batch):
    #     aux_lst = []
    #     for detailed_utterance in batch['phonetic_detail']:
    #         lst = []
    #         for phone in detailed_utterance['utterance']:
    #             lst.append("["+phone+ "]")
    #         detailed_utterance['wrapped_utterance'] = lst[:]
    #         aux_lst.append(detailed_utterance)
    #     batch['phonetic_detail'] = aux_lst[:]
    #     return batch
    # timit = timit.map(wrap_phones_fn, batch_size=-1, keep_in_memory=True, batched = True)
    # timit

    ## CREATE DICTIONARY FROM PHONES FOR ENCODING
    all_phones = []

    def extract_all_phones(batch):
      #This line is the phones of the utterance
      for detailed_utterence in batch["phonetic_detail"]:
        for phone in detailed_utterence['utterance']:
            all_phones.append(phone)
      vocab = list(set(all_phones))
      return {"vocab": [vocab], "all_phones": [all_phones]}


    if not os.path.isfile('./vocab.json'):
        vocabs = timit.map(extract_all_phones, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])

        vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

        vocab_dict = {v: k for k, v in enumerate(vocab_list)}


        ## CONVERT TO UNICODE AND CREATE CORRESPONDING DICT
        # make a copy
        unicode_dict = vocab_dict.copy()
        # reverse the dict
        unicode_dict = {value: key for (key, value) in unicode_dict.items()}
        # make it str to unicode dict
        unicode_to_numeric_dict = {key: chr(0x0001F970+key) for (key, value) in unicode_dict.items()}
        unicode_to_numeric_dict = {value: key for (key, value) in unicode_to_numeric_dict.items()}
        # unicode to numeric dict
        str_to_unicode_dict = {chr(0x0001F970+key): value for (key, value) in unicode_dict.items()}
        str_to_unicode_dict = {value: key for (key, value) in str_to_unicode_dict.items()}


        ## ADD UNK AND PAD
        unicode_to_numeric_dict["[UNK]"] = len(unicode_to_numeric_dict)
        unicode_to_numeric_dict["[PAD]"] = len(unicode_to_numeric_dict)
        unicode_to_numeric_dict["|"] = len(unicode_to_numeric_dict)
        print(len(unicode_to_numeric_dict))

        ## CORRECT FOR UNK AND PAD IN STR DICT
        str_to_unicode_dict["[UNK]"] = "[UNK]"
        str_to_unicode_dict["[PAD]"] = "[PAD]"
        str_to_unicode_dict["|"] = "|"

        ## SAVE DICT TO FILE
        #save unic-numeric (for decoding logits)
        print('writing json files')
        with open('vocab.json', 'w') as vocab_file:
            json.dump(unicode_to_numeric_dict, vocab_file)
        #save str-unicode (decode back to arphabet)
        with open('str_unic.json', 'w') as string_unic_file:
            json.dump(str_to_unicode_dict, string_unic_file)

        # os.system('cp vocab.json /mnt/data/ying/checkpoints/Thesispackage/saves_torgoTD_SSD_1b/vocab.json')
        # os.system('cp str_unic.json /mnt/data/ying/checkpoints/Thesispackage/saves_torgoTD_SSD_1b/str_unic.json')
    else:
        print('loading json files')
        with open('vocab.json') as vocab_file:
            unicode_to_numeric_dict = json.loads(vocab_file.read())
            print("unicode_to_numeric_dict: ",unicode_to_numeric_dict)
        with open('str_unic.json') as str_unic_file:
            str_to_unicode_dict = json.loads(str_unic_file.read())
            print("str_to_unicode_dict: ", str_to_unicode_dict)

    def to_unicode_fn(batch):
        aux_lst = []
        for detailed_utterance in batch['phonetic_detail']:
            lst = []
            for phone in detailed_utterance['utterance']:
                lst.append(str_to_unicode_dict[phone])
            detailed_utterance['unic_utterance'] = lst[:]
            aux_lst.append(detailed_utterance)
        batch['phonetic_detail'] = aux_lst[:]
        return batch

    timit = timit.map(to_unicode_fn, batch_size=-1, keep_in_memory=True, batched=True)
    timit

    ## CONVERT LIST OF PHONES TO STRING OF PHONES
    def delim_phones_fn(batch):
        for detailed_utterance in batch['phonetic_detail']:
            #detailed_utterance['string_utterance'] = '|'.join(detailed_utterance['unic_utterance'])
            detailed_utterance['string_utterance'] = ''.join(detailed_utterance['unic_utterance'])
        return batch
    timit = timit.map(delim_phones_fn, batch_size=-1, keep_in_memory=True, batched = True)
    timit


    ## BUILD PROCESSOR
    #if not (os.path.isdir('./saves') and os.path.isfile('./vocab.json')):
    if True:
        ## TOKENIZER CLASS
        tokenizer = Wav2Vec2CTCTokenizer(vocab_file = "/mnt/data/ying/checkpoints/Torgo/saves_TORGO_TIMIT/vocab.json",
                                         unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

        ## CREATE FEATURE EXTRACTOR
        # Consider adjusting use of hyperparam return_attention_mask.
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=False)

        ## CREATE PROCESSOR
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        ## SAVE PROCESSOR
        processor.save_pretrained("/mnt/data/ying/checkpoints/Torgo/saves_TORGO_TIMIT")
    elif os.path.isdir('./saves_TORGO_TIMIT'):
        processor = Wav2Vec2Processor.from_pretrained("/mnt/data/ying/checkpoints/Torgo/saves_TORGO_TIMIT")


    ### PREPROCESSING
    ## VIEW
    print(timit["train"][0])

    ## CONVERSION TO 1D ARR
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_phones"] = batch['phonetic_detail']['string_utterance']
        return batch

    timit = timit.map(speech_file_to_array_fn, remove_columns=timit.column_names["train"], num_proc=8)

    ## VALIDATE SHAPE
    rand_int = random.randint(0, len(timit["train"]))
    print(rand_int)
    print("Target phones:", timit["train"][rand_int]["target_phones"])
    print("Input array shape:", np.asarray(timit["train"][rand_int]["speech"]).shape)
    print("Sampling rate:", timit["train"][rand_int]["sampling_rate"])

    ## PROCESS

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        # get the audio data
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        # assign the labels
        with processor.as_target_processor():
            # dict = processor.current_processor.decoder
            # reversed_dict = {value: key for (key, value) in dict.items()}
            #
            # tmplist = []
            #
            # for utterence in batch["target_phones"]:
            #     encoded = list()
            #     for phone in utterence:
            #         encoded.append(reversed_dict[phone])
            #
            #     tmplist.append(encoded)
            #
            # #Pass by value
            # batch["labels"] = tmplist[:]
            batch["labels"] = processor(batch["target_phones"], is_split_into_words = False).input_ids

        return batch


    timit_prepared = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], batch_size=8,
                               batched=True)

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

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # wer_metric = load_metric("wer")

    # pred_str = "🥷🦤🦗🦞🦟🦝🦎🦡🦆🦗🥼🦛🦦🦫🥴🦏🦔🥼🥶🥴🦗🦕🦠🥼🦢🦀🦇🦐🥻🦨🦦🦂🦉🦇🦓🦕🦇🦌🦤🦫🥵🥰🦌🥷"
    # label_str = "🥱🦍🦁🥷🦔🦒🥲🦘🦁🦅🦩🦤🦁🦬🦣🥰🦅🥾🦬🦁🦚🥴🦅🥵🦀🦗🦋🦪🦐🦤🦖🦇🦠🦛🦚🦗🦓🦍🦠🦢🥺🥳🥱"
    def character_error_rate(pred_str, label_str):
        preds = [char for seq in pred_str for char in list(seq)]
        refs = [char for seq in label_str for char in list(seq)]
        error = wer(refs, preds)
        return error

    # preds = ['\U0001f977', '\U0001f9a4', '🦗', '🦞', '🦟', '🦝', '🦎', '🦡', '🦆', '🦗', '🥼', '🦛', '\U0001f9a6', '\U0001f9ab', '🥴', '🦏', '🦔', '🥼', '🥶', '🥴', '🦗', '🦕', '🦠', '🥼', '🦢', '🦀', '🦇', '🦐', '\U0001f97b', '\U0001f9a8', '\U0001f9a6', '🦂', '🦉', '🦇', '🦓', '🦕', '🦇', '🦌', '\U0001f9a4', '\U0001f9ab', '🥵', '🥰', '🦌','\U0001f977']
    # refs = ['\U0001f971', '🦍', '🦁', '\U0001f977', '🦔', '🦒', '\U0001f972', '🦘', '🦁', '🦅', '\U0001f9a9', '\U0001f9a4', '🦁', '\U0001f9ac', '\U0001f9a3', '🥰', '🦅', '🥾', '\U0001f9ac', '🦁', '🦚', '🥴', '🦅', '🥵', '🦀', '🦗', '🦋', '\U0001f9aa', '🦐', '\U0001f9a4', '🦖', '🦇', '🦠', '🦛', '🦚', '🦗', '🦓', '🦍', '🦠', '🦢', '🥺', '🥳', '\U0001f971']

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_ids
        pred_str = processor.batch_decode(pred_ids)
        #print(pred_str)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        error = character_error_rate(pred_str, label_str)

        #phone_edit_dist = wer_metric.compute(predictions=" ".join(pred_str), references=" ".join(label_str))

        # {"phone_edit_dist": phone_edit_dist}

        return {"cer": error}


    # model = Wav2Vec2ForCTC.from_pretrained(
    #     "facebook/wav2vec2-large-xlsr-53",
    #     gradient_checkpointing=True,
    #     ctc_loss_reduction="mean",
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     vocab_size=len(processor.tokenizer),
    # )

    # /mnt/data/ying/checkpoints/Thesispackage_Torgo/wav2vec2-xls-r-1b-torgoTD/8bs50e-1b-1e-5/checkpoint-8000
    # /mnt/data/ying/checkpoints/Thesispackage/wav2vec2-1b-timit-torgoTD/8bs50e-1e-5/checkpoint-4750(CER:0.081)
    config = Wav2Vec2Config.from_pretrained("/mnt/data/ying/checkpoints/Torgo/wav2vec2_1b_TORGO/8bs50e_1b_1e-5/checkpoint-4250")
    config.vocab_size = VOCAB_SIZE

    model = Wav2Vec2ForCTC.from_pretrained(
        "/mnt/data/ying/checkpoints/Torgo/wav2vec2_1b_TORGO/8bs50e_1b_1e-5/checkpoint-4250",
        ignore_mismatched_sizes=True,
        # gradient_checkpointing=True,
        # ctc_loss_reduction="mean",
        # pad_token_id=processor.tokenizer.pad_token_id,
        # vocab_size=len(processor.tokenizer),
    )

    model.gradient_checkpointing_enable()
    model.pad_token_id=processor.tokenizer.pad_token_id
    model.ctc_loss_reduction="mean"
    model.vocab_size=len(processor.tokenizer)

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
      # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
      output_dir="/mnt/data/ying/checkpoints/Torgo/wav2vec2_1b_TORGO_TIMIT/8bs50e_1e-5",
      group_by_length=True,
      gradient_checkpointing=True,
      per_device_train_batch_size=8,
      evaluation_strategy="steps",
      num_train_epochs=50,
      fp16=True,
      save_steps=250,
      eval_steps=250,
      logging_steps=500,
      learning_rate=1e-5,
      weight_decay=0.005,
      warmup_steps=1000,
      save_total_limit=2,
      load_best_model_at_end = True,
      report_to="wandb",  # enable logging to W&B
      run_name=datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # name of the W&B run (optional)
    #   run_name = "newphonetest-torgo-dataset-Bryce-model-just-edit-paths-after-18Aug-meeting"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit_prepared["train"],
        eval_dataset=timit_prepared["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # def map_to_result(batch):
    #     model.to("cuda")
    #     input_values = processor(
    #         batch["speech"],
    #         sampling_rate=batch["sampling_rate"],
    #         return_tensors="pt"
    #     ).input_values.to("cuda")

    #     with torch.no_grad():
    #         logits = model(input_values).logits

    #     pred_ids = torch.argmax(logits, dim=-1)
    #     batch["pred_str"] = processor.batch_decode(pred_ids)[0]

    #     return batch

    # processor = Wav2Vec2Processor.from_pretrained("/mnt/data/ying/checkpoints/Thesispackage/saves_timit_torgoTD_SSD_1b")
    # model = Wav2Vec2ForCTC.from_pretrained("/mnt/data/ying/checkpoints/Thesispackage/wav2vec2-1b-timit-torgoTD-SSD/8bs50e-1e-5/checkpoint-9750")
    # results = timit["test"].map(map_to_result)
    # # print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["target_phones"])))

    # def average_character_error_rate(pred_list, label_list):
    #     print("************** average_character_error_rate ***********")
    #     errList = list()
    #     assert len(pred_list) == len(label_list), "Prediction list and label list must be of equal length"

    #     for index in range(len(pred_list)):
    #         errList.append(character_error_rate(pred_list[index], label_list[index]))

    #     total = sum(errList)
    #     avg = total/len(pred_list)

    #     return avg

    # print("Test WER: {:.3f}".format(average_character_error_rate(pred_list=results["pred_str"], label_list=results["target_phones"])))


    # def pred_single_wav(ckpt, wavpath):
    #     try:
    #         if ckpt != None:
    #             model = Wav2Vec2ForCTC.from_pretrained(ckpt)
    #         else:
    #             model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b")

    #         model.to("cuda")

    #         # load wav
    #         wav_arr = sf.read(wavpath)

    #         input_values = processor(wav_arr[0], sampling_rate=wav_arr[1],
    #                                       return_tensors="pt").input_values.to("cuda")

    #         with torch.no_grad():
    #             logits = model(input_values).logits

    #         pred_ids = torch.argmax(logits, dim=-1)

    #         # convert ids to tokens
    #         unicode_to_str_dict = {value: key for (key, value) in str_to_unicode_dict.items()}
    #         tokens = processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())
    #         print(tokens)
    #         phones = ["".join(unicode_to_str_dict[token]) for token in tokens]
    #         print(phones)
    #     except Exception as e:
    #         print(e)

    # pred_single_wav('/mnt/data/ying/checkpoints/Thesispackage/wav2vec2-1b-timit-torgoTD-SSD/8bs50e-1e-5/checkpoint-9750', '/mnt/data/ying/TORGO_Control_Group_Amend_PHN_SN_divided_same_as_SSD_group/data/TEST/MC04/Session1/0005.wav')



if __name__ == "__main__":
    main()


