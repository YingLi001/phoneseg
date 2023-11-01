import pandas as pd

# import utilities
from scipy.io import wavfile
from transformers.utils.dummy_pt_objects import torch_distributed_zero_first
import os
import w2v2_predict
# VAD is from here: @author: eesungkim
from utils.vad import *

from UnsupSeg import predict as seg_pred

from datasets import load_dataset, load_metric

import newphonetest as npt

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

import sys

import pickle
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics

import soundfile as sf
from multiprocessing.dummy import Pool as ThreadPool
from datetime import date, datetime
import pytz
from praatio import textgrid

bias = 0.5
use_vad = False
#soft clean
use_clean = True
# #False for midpoints, True for onset boundaries
# AIE_evaluation = False
#hard clean
clean_aggressive = True

#   Of an array of 0's and 1's, return the time in seconds when the transitions occur
#   arr: an array of zeroes and ones
#   st: sample length in time (s)
def detectEdges(arr, st):
    tmp = None
    tmp2 = None
    return_array = list()

    for ii in range(len(arr)):
        jj = int(ii)
        if tmp == None:
        #Condtion: we just instantiated the array
            tmp = arr[jj][0]

        #Condition: rising edge
        elif arr[jj][0] != tmp and tmp == 0:
            #Store timing of rising edge in seconds
            tmp2 = jj * st/len(arr)

            tmp = 1

        # Condition: falling edge
        elif arr[jj][0] != tmp and tmp == 1:
            #add the pair of boundaries to the return list
            return_array.append((tmp2, jj * st/len(arr)))
            tmp = 0

    return return_array

#   removes segmentations which are deemed to have
#   occured in spaces of audio where there is no speech
#
#   Signal:     signal data to process
#   sr:         sampling rate
#   tolerance: tolerance in difference in VAD boundaries and Seg boundaries difference
def filterSegmentations(segmentations, signal, sr, tolerance = 0.05):
    vad=VAD(signal, sr, nFFT=2048, win_length=0.025, hop_length=0.01, theshold=0.5)
    vad = vad.astype('int')
    vad = vad.tolist()
    vadEdges = detectEdges(vad, len(signal)/sr)

    filtered_segs = list()
    # Scan through all the segmentations looking for segments which fit withing the boundaries.
    # Works in O(N^2) time because im a pig.
    #filtered_segs.append(0)
    for seg in segmentations:
        for vadBound in vadEdges:
            if vadBound[0]-tolerance <= seg and seg <= vadBound[1]+tolerance:
                filtered_segs.append(seg)
    #filtered_segs.append(len(signal)/sr)

    return filtered_segs

def filterSegmentationsWrapper(wav_path, segmentations):
    signal, sr  = sf.read(wav_path)
    return filterSegmentations(segmentations, signal, sr)

def seg_demo(wav_path, vad = use_vad, clean = use_clean, clean_agg = clean_aggressive):
    all_alignment_info = []
    signalData, samplingFrequency  = sf.read(wav_path)

    #Duration of utterance in seconds
    seconds = len(signalData)/samplingFrequency
    print(seconds)
    wp = w2v2_predict.w2v2_predictor()
    wp.set_model(ckpt="/mnt/data/ying/checkpoints/Torgo/wav2vec2_1b_TORGO/8bs50e_1b_1e-5/checkpoint-4250")

    #tokens = ['h#', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'hh', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', 'l', '[PAD]', '[PAD]', 'ow', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'pau', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'dh', 'dh', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'q', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', 'z', '[PAD]', '[PAD]', 'ix', '[PAD]', '[PAD]', 'tcl', 'tcl', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'tcl', '[PAD]', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'h#']
    tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")
    print('tokens (no collapsed)', tokens)
    collapsed_tokens = wp.pred_wav_with_collapse(wav_path, return_type="phones")
    print('collasped tokens', collapsed_tokens)
    #Use Unsupseg38 by felix kreuk to get the segmentations
    # /mnt/data/ying/UnsupSeg_experiments/torgoTD/unsupervised_segmentor/2023-04-20_22-11-37-cosine_coef_0.6_2nd/epoch=2.ckpt
    # /home/ying/Thesispackage_Torgo/UnsupSeg/runs/unsupervised_segmentor/2023-02-21_11-47-24-default/epoch=6.ckpt
    
    # # For SegFeat
    # segfeat_intervals = []
    # segVect = []
    # with open("/mnt/data/ying/TORGO_Control_Group_Amend_PHN_SN_divided_same_as_SSD_group/data/TRAIN/MC03/Session2/0008_SegFeat_127ckpt_61dict_result.json", "r") as fp:   
    #     info = json.load(fp)
    #     segfeat_intervals = info[0]['data']

    # i = 0
    # for item in segfeat_intervals:
    #     i = i + 1
    #     if i == len(segfeat_intervals):
    #         segVect.append(item[0]/100)
    #         segVect.append(item[1]/100)
    #     else:
    #         segVect.append(item[0]/100)

    # For UnsupSeg
    segVect = seg_pred.pred(wav=wav_path, ckpt='/home/ying/Thesispackage_Torgo/UnsupSeg/runs/unsupervised_segmentor/2023-02-21_11-47-24-default/epoch=6.ckpt', prominence=None)
    if vad == True:
        print("Use VAD")
        segVect = filterSegmentationsWrapper(wav_path=wav_path,
                                                segmentations=segVect)
    else:
        print("Not Use VAD")
        segVect = segVect.tolist()
    
    print('segVect (SegFeat)', segVect)
    all_alignment_info.append(segVect)
    all_alignment_info.append(tokens)
    all_alignment_info.append(collapsed_tokens)
    #Delta s is half the distance in time between each token
    delta_s = seconds / (2*len(tokens))

    #A list of tokens with time attached. It's called votelist because itll do some voting later on
    timed_token_list = list()

    #instantiate timestamp with one delta s. The distance between each token in time is 2 times delta_s
    timestamp = delta_s

    #This for loop creats a list of tuples with the timing attached to each
    for token in tokens:
        #Timed token is a tuple with the time in the sequence at which it occurs
        timed_token = (token, timestamp)

        #Add timed token to the voter list
        timed_token_list.append(timed_token)

        #Increment the timestamp for the next token
        timestamp = timestamp + 2*delta_s

    # Now keep only the labels worth interpreting
    print("timed_token_list",timed_token_list)
    filtered_time_token_list = list()
    for tt in timed_token_list:
        if tt[0] != ("[PAD]" or "[UNK]" or "|"):
            filtered_time_token_list.append(tt)
    print("=============== filtered_time_token_list ===============")        
    print(filtered_time_token_list)
    print("======================================================")
    all_alignment_info.append(filtered_time_token_list)
    # Compute Decision Boundaries
    def decision_boundary_calc(filtered_time_token_list, seconds, bias=bias):
        print("Bias:", bias )
        assert 0 <= bias and bias <= 1
        DCB = list()
        for ii in range(len(filtered_time_token_list)):
            if ii == len(filtered_time_token_list) - 1:  # CASE: Last token
                upper = seconds
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            elif ii == 0:  # CASE: First token
                upper = filtered_time_token_list[ii + 1][1] * (bias) + filtered_time_token_list[ii][1] * (1 - bias)
                lower = 0
            else:
                upper = (filtered_time_token_list[ii + 1][1]) * (bias) + (filtered_time_token_list[ii][1]) * (1 - bias)
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            # append phone label, start time, end time tuple
            DCB.append((filtered_time_token_list[ii][0], lower, upper))
        return DCB

    DCB = decision_boundary_calc(filtered_time_token_list, seconds)
    all_alignment_info.append(DCB)
    print("======================= DCB ==========================")
    print(DCB)
    print("======================================================")
    #Assign Maximal labels
    try:
        with open('str_unic.json') as str_unic_file:
            str_to_unicode_dict = json.loads(str_unic_file.read())
        Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict,  0)
    except Exception:
        print("Failed to open str_unic.json")

    # Uncomment the following lines for UnsupSeg and comment for SegFeat  
    # insert '0' at 0 index (first 0 is index and the second 0 is the inserted content)
    segVect.insert(0, 0)
    segVect.append(seconds)

    label_list = list()
    
    for segIndex in range(len(segVect)):
        label_dict = Max_DCB_init_dict.copy()

        if segIndex != (len(segVect) - 1):
            t_segStart = segVect[segIndex]
            t_segEnd = segVect[segIndex+1]
            for dcb in DCB:
                #CASE: decision starts within the segment
                if  t_segStart <= dcb[1] and dcb[1] <= t_segEnd:
                    #CASE: Decision contained entirely within the segment
                    if dcb[2] <= t_segEnd :
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - dcb[1])
                    else: #CASE: Decision starts within the segment but ends elsewhere
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - dcb[1])
                #CASE: Decision ends within the segment, but does not start within the segmnet
                elif t_segStart  <= dcb[2] and dcb[2] <= t_segEnd:
                    label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - t_segStart)
                #CASE: Decision contains the entirety of the seg
                elif  dcb[1] <=  t_segStart and t_segEnd <= dcb[2] :
                    label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - t_segStart)
                else:
                    pass
                #dcb[0] : phone label. dcb[1] : start time, dcb[2] : end time

            label_list.append(max(label_dict, key=label_dict.get))
    all_alignment_info.append(label_list)

    #Lets zip each label with its start and end times
    segList = list()
    [segList.append((label_list[ii], segVect[ii], segVect[ii+1])) for ii in range(len(label_list))]
    print("============== before cleaning ===================")
    print(segList)
    print("==================================================")

    def clean_segs(segList_in, wav_path):
        segList = segList_in.copy()
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)

        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))

        index = 0
        for jj in range(len(transitions)):
            found = False
            limitreached = False

            while found == False and limitreached == False:
                if index >= len(segList)-1:
                    limitreached = True
                else:
                    seg_from = segList[index]
                    seg_to = segList[index + 1]

                    #CASE: the two elements are the same, ie seglist: aab, transition: ab, focal:aa, turn seglist into ab
                    if seg_from[0] == seg_to[0] and seg_from[0] == transitions[jj][0]:
                        segList[index] = (segList[index][0], segList[index][1], segList[index+1][2])
                        segList.remove(segList[index+1])


                    #CASE: Transition is found
                    elif seg_from[0] == transitions[jj][0] and seg_to[0] == transitions[jj][1]:
                        found = True
                        index = index + 1

                    else:
                        index = index + 1
                        break
        return segList

    def clean_segs_aggressive(segList_in, wav_path):
        print("Hard Clean Start")
        segList = segList_in.copy()
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)
        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))

        ceiling = len(segList)-2
        jj=0
        finished = False
        while finished == False:
            if jj <= ceiling:
                if segList[jj][0]==segList[jj+1][0]:
                    if not (segList[jj][0],segList[jj+1][0]) in transitions:
                        newSeg = (segList[jj][0], segList[jj][1], segList[jj+1][2])
                        segList[jj] = newSeg
                        segList.remove(segList[jj+1])
                        ceiling = ceiling-1
                    jj = jj - 1
                jj = jj + 1
            else:
                finished = True
        print("Hard Clean Finished")
        return segList
    
    if clean == True:
            print("===== checking the cleaning method =====")
            if clean_agg == True:
                print("===== Use Hard Clean======")
                segList = clean_segs_aggressive(segList, wav_path)
                
            else:
                print("====== Use Soft Clean ======")
                segList = clean_segs(segList, wav_path)

    # segList = clean_segs_aggressive(segList, wav_path)

    temp = list()
    segList_str = []
    for seg in segList:
        temp.append((seg[1],seg[2],seg[0]))
        segList_str.append(str(seg[1])+'-'+str(seg[2])+'-'+seg[0])
    segList = temp
    all_alignment_info.append(segList)
    with open("/mnt/data/ying/TORGO/data/TEST/M05/Session1/0017_all_alignment_info_1b_TORGO_combined.json", "w") as fp:   
        json.dump(all_alignment_info, fp)
    print('final results', segList)
    return segList

if __name__ == "__main__":
    # wav_path = "/mnt/data/ying/SMAAT_Data_word_level/308 Greyson Audio 1.wav16ksr.wavword1.wav"
    wav_path = "/mnt/data/ying/TORGO/data/TEST/M05/Session1/0017.wav"
    segList_res = seg_demo(wav_path)
    print('final results', segList_res)
    
    # # segList_new = list()
    # # [segList_new.append((segList[ii][1], segList[ii][2], segList[ii][0])) for ii in range(len(segList))]

    # # print(segList_new)
    # tg = textgrid.Textgrid()
    # phonemeTier = textgrid.IntervalTier('phoneme', segList, 0, segList[-1][1])
    # tg.addTier(phonemeTier)
    # tg.save("/mnt/data/ying/SMAAT_Data_word_level/308_Greyson_word1_1Tier.TextGrid", format="short_textgrid", includeBlankSpaces=False)
    # print("Finish generating TextGrid file !!!")
