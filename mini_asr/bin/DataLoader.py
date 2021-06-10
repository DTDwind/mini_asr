#! /usr/bin/python
# -*- encoding: utf-8 -*-

# Copyright 2021 National Taiwan Normal University SMIL. (Author: Yu-Sen Cheng)
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

import torchaudio
import numpy as np
import random
import os
import threading

import time
import math
import glob
from mini_asr.bin import feature
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import kaldi_io

# def round_down(num, divisor):
#     return num - (num%divisor)

def pad_packed_collate(batch, device=torch.device("cpu")):
    """
    Return:
        xs_pad: [B, T, D]
        ys_pad: [B, T, S]
        ilen: lenght of each sequence
        n_speakers: speaker number of each sequence
    """
    # TODO: so dirty...
    ilen = torch.as_tensor([ egs[0].shape[0] for egs in batch ], device=device)

    n_speakers = [ egs[1].shape[1] for egs in batch ]
    max_n_speaker = np.max(n_speakers)
    n_speakers = torch.as_tensor(n_speakers)
    
    xs_batch, ys_batch, reco_batch = [list(samples) for samples in zip(*batch)]
    
    xs_batch = [torch.as_tensor(np.array(xs)) for xs in xs_batch]

    # padding speaker Label
    ys_batch = [ F.pad( torch.as_tensor(ys, dtype=torch.float), (0, max_n_speaker-ys.shape[1]), "constant", 0) 
                 for ys in ys_batch ]

    # padding Label sequence
    ys_pad = torch.nn.utils.rnn.pad_sequence(ys_batch, batch_first=True, padding_value=-1)
    
    # padding input sequence
    xs_pad = torch.nn.utils.rnn.pad_sequence(xs_batch, batch_first=True, padding_value=0)
    return xs_pad, ys_pad, ilen, n_speakers, reco_batch

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_data_loader(data_dir, args):

    train_dataset = naive_kaldi_loader(data_dir, args.max_frames)
    sampler       = DistributedSampler(train_dataset, shuffle=True, seed= args.seed)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=pad_packed_collate,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    return train_loader, sampler

class naive_kaldi_loader(Dataset):
    def __init__(self, data_dir, max_frames):
        """
            Note: feat->(n_frames, n_bins), label->(n_frmaes, n_speakers)
        """

        self.reco_list = []
        self.reco_path = dict()
        self.feats = dict()
        self.labels = dict()
        
        # non-segment so 'uttID == reco'
        # print("data_dir: "+str(data_dir))

        # TODO: reco2dur utt2dur?
        self.utt2dur = self.load_utt2dur(os.path.join(data_dir, 'reco2dur')) # uttID: dur

        # recID: st dur spk  ex: EN2001a:{st:11.09, dur:4.44, spk:MEE068 }
        print(data_dir)
        self.rttm, self.n_speakers, self.speaker_list = self.load_rttm(os.path.join(data_dir, 'rttm.annotation')) 
        
        # for reco, feat in kaldi_io.read_mat_scp(os.path.join(data_dir, 'feats.scp')): # get fbank feat and label
        
        
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_fft=256, win_length=200, hop_length=80, window_fn=torch.hamming_window, n_mels=23)
        # self.stft = torch.stft(input, 256, hop_length=80, win_length=200, window= torch.hann_window())
        for line in open(os.path.join(data_dir, 'wav.scp')):
            reco = line.split()[0]
            path = line.split()[1]

            self.reco_path[reco] = path
            # self.labels[reco] = label
            self.reco_list.append(reco)

        # self.data = np.load('/share/nas167/chengsam/SpeakerDiarization/data_feat/train_960_librispeech_egs_dict.npy', 
        #                              allow_pickle=True).item()

    def __getitem__(self, indices): # Map-style datasets

        reco  = self.reco_list[indices]
        path  = self.reco_path[reco]

        waveform, sample_rate = torchaudio.load(path)
        # torch.set_printoptions(precision=10)
        # print("reco: "+str(reco))
        # print("waveform shape: "+str(waveform.shape))
        # print("waveform numpy shape: "+str(waveform.numpy().shape))
        # print("waveform: "+str(waveform))

        # QAQ = torch.stft(waveform, 256, hop_length=80, win_length=200)


        feat = feature.stft(waveform.numpy()[0], 200, 80)
        feat = feature.transform(feat, "logmel23_mn")
        
        
        
        # print("feat: "+str(torch.as_tensor(feat)))
        # Y_spliced = feature.splice(Y, self.context_size)
        # print("QAQ: "+str(QAQ))
        # quit()
        # feat = self.torchfb(waveform).squeeze()
        # feat = feat.permute(1, 0).numpy()
        # feat = np.maximum(feat, 1e-10)
        # feat = np.log10(np.maximum(feat, 1e-10))
        # feat = feat.log() # log mel
        # print("feat shape: "+str(feat.shape))
        # quit(0)
        
        n_frame = feat.shape[0]
        
        # quit()
        # feat = feat.transpose(1, 0).numpy()
        # # print("utt2dur[reco] : "+str(self.utt2dur[reco]))
        label = np.zeros((n_frame, self.n_speakers[reco]), dtype=np.int32)
        sec_per_frame = round(self.utt2dur[reco]/n_frame, 5) # 0.01 sec
        # print("sec_per_frame: "+str(sec_per_frame))
        # quit(0)
        

        for seg in self.rttm[reco]:

            start_frame = np.int( seg['st'] / sec_per_frame)
            shift_frame = np.int( seg['dur'] / sec_per_frame)
            label[start_frame: (start_frame+shift_frame), self.speaker_list[reco][seg['spk']] ] = 1
        
        # splice
        feat_s = self.splice(feat, 7)
        # subsample
        feat_ss = feat_s[::10]
        label_s = label[::10]
        
        # torch.set_printoptions(edgeitems=50)
        # print("reco: "+str(reco))
        # print("feat_s: "+str(torch.as_tensor(feat_s)))
        # print("label_s size: "+str(label_s.shape))
        # quit()
        
        
        # label_ss = self.splice(label_s, 7)

        return feat_ss, label_s, reco

        
        # feat = self.data[reco][0]
        # label = self.data[reco][1]

        # On the fly feature extract

        # TODO: shuffle
        # if self.shuffle:
        # order = np.arange(label.shape[0])
        # np.random.shuffle(order)
        # feat = feat[order]
        # label = label[order]

        # return feat, label, reco

    def splice(self, Y, context_size=7):
        """ Frame splicing

        Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
        Licensed under the MIT license.
        
        Args:
            Y: feature
                (n_frames, n_featdim)-shaped numpy array
            context_size:
                number of frames concatenated on left-side
                if context_size = 5, 11 frames are concatenated.

        Returns:
            Y_spliced: spliced feature
                (n_frames, n_featdim * (2 * context_size + 1))-shaped
        """
        Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
        Y_spliced = np.lib.stride_tricks.as_strided(
            np.ascontiguousarray(Y_pad),
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
        return Y_spliced

    def __len__(self):
        return len(self.reco_list)

    def load_utt2dur(self, utt2dur_file):
        """ returns dictionary { recid: duration }  """

        if not os.path.exists(utt2dur_file):
            return None

        lines = [line.strip().split(None, 1) for line in open(utt2dur_file)]

        return {x[0]: float(x[1]) for x in lines}

    def load_rttm(self, rttm_file):
        
        ret = {}
        n_speakers = {}
        speaker_list = {}

        if not os.path.exists(rttm_file):
            return None

        for line in open(rttm_file):

            tag_type, fileID, chID, st, dur, na, na, spk, na  = line.strip().split()

            # E.g. ['SPEAKER', 'EN2001a', '1', '3.34', '0.54', '<NA>', '<NA>', 'MEO069', '<NA>', '<NA>']
            if fileID not in ret: # init
                ret[fileID] = []
                n_speakers[fileID] = 0
                speaker_list[fileID] = {}

            if spk not in speaker_list[fileID]:
                speaker_list[fileID][spk] = n_speakers[fileID]
                n_speakers[fileID] += 1
            ret[fileID].append({'st':float(st), 'dur':float(dur), 'spk':spk})

        return ret, n_speakers, speaker_list

class test_dataset_loader(Dataset):
    def __init__(self, data_dir, subsampling):
        """
            Note: feat->(n_frames, n_bins)
        """
        self.subsampling = subsampling
        self.reco_list = []
        self.feats = dict()
        
        # for reco, feat in kaldi_io.read_mat_scp(os.path.join(data_dir, 'feats.scp')): # get fbank feat and label
                
        #     # self.feats[reco] = feat
        #     self.reco_list.append(reco)
        # self.data = np.load('/share/nas167/chengsam/SpeakerDiarization/data_feat/dev_egs_dict.npy', 
        #                              allow_pickle=True).item()
        self.reco_path = dict()
        # self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_fft=256, win_length=200, hop_length=80, window_fn=torch.hamming_window, n_mels=23)
        
        for line in open(os.path.join(data_dir, 'wav.scp')):
            reco = line.split()[0]
            path = line.split()[1]

            self.reco_path[reco] = path
            self.reco_list.append(reco)


    def __getitem__(self, indices): # Map-style datasets

        reco  = self.reco_list[indices]
        path  = self.reco_path[reco]

        waveform, sample_rate = torchaudio.load(path)
        feat = feature.stft(waveform.numpy()[0], 200, 80)
        feat = feature.transform(feat, "logmel23_mn")
        
        n_frame = feat.shape[0]

        feat_s = self.splice(feat, 7)
        feat_ss = feat_s[::10]
        ilen = torch.as_tensor(feat_ss.shape[0])
        return torch.from_numpy(feat_ss), ilen, reco


    def __len__(self):
        return len(self.reco_list)

    def splice(self, Y, context_size=7):
        """ Frame splicing

        Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
        Licensed under the MIT license.
        
        Args:
            Y: feature
                (n_frames, n_featdim)-shaped numpy array
            context_size:
                number of frames concatenated on left-side
                if context_size = 5, 11 frames are concatenated.

        Returns:
            Y_spliced: spliced feature
                (n_frames, n_featdim * (2 * context_size + 1))-shaped
        """
        Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
        Y_spliced = np.lib.stride_tricks.as_strided(
            np.ascontiguousarray(Y_pad),
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
        return Y_spliced
