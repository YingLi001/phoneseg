'''
@Course name: HDR
@Project   : SMAAT
@File      : torgo.py
@Author    : Ying Li
@Student ID: 20909226
@Year      : 2022
'''
# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
from glob import glob
import re
import random

"""TORGO automatic speech recognition dataset."""

import os
from pathlib import Path

import datasets
from datasets.tasks import AutomaticSpeechRecognition

_CITATION = """\
@article{Rudzicz2012TheTD,
  title={The TORGO database of acoustic and articulatory speech from speakers with dysarthria},
  author={Frank Rudzicz and Aravind Kumar Namasivayam and Talya Wolff},
  journal={Language Resources and Evaluation},
  year={2012},
  volume={46},
  pages={523-541}
}
"""

_DESCRIPTION = """\
The TORGO database of dysarthric articulation consists of aligned acoustics and measured 3D articulatory features 
from speakers with either cerebral palsy (CP) or amyotrophic lateral sclerosis (ALS), which are two of the most 
prevalent causes of speech disability (Kent and Rosen, 2004), and matchd controls. This database, called TORGO, 
is the result of a collaboration between the departments of Computer Science and Speech-Language Pathology 
at the University of Toronto and the Holland-Bloorview Kids Rehab hospital in Toronto.
More info on TORGO dataset can be understood from the "README" which can be found here:
http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
"""

_HOMEPAGE = "http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html"


class TORGOASRConfig(datasets.BuilderConfig):
    """BuilderConfig for TimitASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(TORGOASRConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class TORGOASR(datasets.GeneratorBasedBuilder):
    """TimitASR dataset."""

    BUILDER_CONFIGS = [TORGOASRConfig(name="clean", description="'Clean' speech.")]

    @property
    def manual_download_instructions(self):
        return (
            "To use TORGO you have to download it manually. "
            "Please create an account and download the dataset from http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html \n"
            "Then extract all files in one folder and load the dataset with: "
            "`datasets.load_dataset('torgo', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "phonetic_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                    # "word_detail": datasets.Sequence(
                    #     {
                    #         "start": datasets.Value("int64"),
                    #         "stop": datasets.Value("int64"),
                    #         "utterance": datasets.Value("string"),
                    #     }
                    # ),
                    # "dialect_region": datasets.Value("string"),
                    # "sentence_type": datasets.Value("string"),
                    # "speaker_id": datasets.Value("string"),
                    # "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            # task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('torgo', data_dir=...)` that includes files unzipped from the TIMIT zip. Manual download instructions: {self.manual_download_instructions}"
            )

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"split": "TRAIN", "data_dir": data_dir}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"split": "TEST", "data_dir": data_dir}),
        ]

 
    #     return phn_paths
    def _generate_examples(self, split, data_dir):
        
        """Generate examples from TIMIT archive_path based on the test/train csv information."""
        # Iterating the contents of the data to extract the relevant information
        wav_paths = sorted(Path(data_dir).glob(f"**/{split}/**/**/*.wav"))
        # print("wav_paths-1",wav_paths)
        wav_paths = wav_paths if wav_paths else sorted(Path(data_dir).glob(f"**/{split.upper()}/**/**/*.WAV"))
        # print("wav_paths",wav_paths)
        s = set()
        for key, wav_path in enumerate(wav_paths):
            # extract transcript
            txt_path = with_case_insensitive_suffix(wav_path, ".txt")
            # print("txt_path",txt_path)
            with txt_path.open(encoding="utf-8") as op:
                
                transcript = " ".join(op.readline().split())

            # extract phonemes
            phn_path = with_case_insensitive_suffix(wav_path, ".phn")
            # print("phn_path",phn_path)
            with phn_path.open(encoding="utf-8") as op:
                    phonemes = [
                        {
                            "start": i.split(" ")[0],
                            "stop": i.split(" ")[1],
                            "utterance": " ".join(i.split(" ")[2:]).strip(),
                        }  
                        for i in op.readlines()
                    ]
            # extract words
            # wrd_path = with_case_insensitive_suffix(wav_path, ".wrd")
            # with wrd_path.open(encoding="utf-8") as op:
            #     words = [
            #         {
            #             "start": i.split(" ")[0],
            #             "stop": i.split(" ")[1],
            #             "utterance": " ".join(i.split(" ")[2:]).strip(),
            #         }
            #         for i in op.readlines()
            #     ]

            # dialect_region = wav_path.parents[1].name
            # sentence_type = wav_path.name[0:2]
            # speaker_id = wav_path.parents[0].name[1:]
            # id_ = wav_path.stem

            example = {
                "file": str(wav_path),
                "audio": str(wav_path),
                "text": transcript,
                "phonetic_detail": phonemes,
                # "word_detail": words,
                # "dialect_region": dialect_region,
                # "sentence_type": sentence_type,
                # "speaker_id": speaker_id,
                # "id": id_,
            }

            yield key, example


def with_case_insensitive_suffix(path: Path, suffix: str):
    path = path.with_suffix(suffix.lower())
    path = path if path.exists() else path.with_suffix(suffix.upper())
    return path
