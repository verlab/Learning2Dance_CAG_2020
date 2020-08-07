#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np
import librosa
import pickle
import scipy
import torch
import torchaudio
import os

import pdb

class pose_audio_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, sample_size=64, stride=16, data_aug=0, create_z=False, sample_rate=16000, keep_wav=True, styles_to_remove=[], pre_process=True):

        print("Initializing dataset...")

        ## LOADING METADATA
        metadata_file = dataset_path + 'metadata_' + str(sample_size) + '_' + str(stride) + '_' + str(data_aug) + '.pickle'

        ## Check if file exists, if not, then create metadata.
        if not os.path.exists(metadata_file):
            print("Creating metadata file...")
            try:
                cmd = "python create_metadata_toy.py --dataset_path " + dataset_path + " --sample_size " + str(sample_size) + \
                    " --data_aug " + str(int(data_aug)) + " --stride " + str(stride)
                os.system(cmd)
            except:
                cmd = "python3 create_metadata_toy.py --dataset_path " + dataset_path + " --sample_size " + str(sample_size) + \
                    " --data_aug " + str(int(data_aug)) + " --stride " + str(stride)
                os.system(cmd)

        with open(metadata_file, 'rb') as handle:
            self.metadata_dict = pickle.load(handle)

        ## REMOVE UNWANTED STYLES
        for style_to_remove in styles_to_remove:
            self.metadata_dict = self.remove_style(self.metadata_dict, style_to_remove)

        ## CLASS VARIABLES 
        self.dataset_length = len(self.metadata_dict)
        self.styles = sorted(list(set([self.metadata_dict[sample]['style'] for sample in self.metadata_dict])))
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.keep_wav = keep_wav
        self.z_path = dataset_path + "z_data_" + str(sample_size) + '_' + str(stride) + '_' + str(data_aug) + '/'
        self.wavs_dict = {}
        self.num_styles = len(self.styles)
        self.styles_dict = dict(zip(range(self.num_styles),self.styles))
        self.pre_process = pre_process

        ## STORING WAV FILES IN MEMORY, IF SPECIFIED
        if self.keep_wav:
            audios_paths = list(set([self.metadata_dict[sample]['audio_path'] for sample in self.metadata_dict]))

            for audio_path in audios_paths:
                audio, sr = torchaudio.load(audio_path)
                if self.pre_process:
                        audio = torchaudio.functional.mu_law_encoding(audio, 16)
                #audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                self.wavs_dict[audio_path] = audio

        ## CREATE AND SAVE Z
        if not os.path.exists(self.z_path) or create_z:
            os.makedirs(self.z_path, exist_ok=True)

            for idx in tqdm(range(self.dataset_length), desc="Generating Z data..."):
                z = self.make_z(idx)
                np.save(self.z_path + str(idx) + "_z.npy", z)

        print("Done.")

    def __len__(self):
        return(self.dataset_length)

    def __getitem__(self, idx):
        try:
            ## SAMPLE VARS
            sample_dict = self.metadata_dict[idx]

            sample_skeletons =       sample_dict['samples_frames']
            sample_audio_intervals = sample_dict['samples_audio']
            sample_audio_path =      sample_dict['audio_path']
            sample_style =           sample_dict['style']

            ## LOAD POSES
            skeletons = np.empty( (self.sample_size, 25, 2) )
            skeletons = np.array([np.load(skeleton_path) for skeleton_path in sample_skeletons])

            ## LOAD AUDIO
            if self.keep_wav:
                audio = self.wavs_dict[sample_audio_path]
            else:
                # pdb.set_trace()
                audio, sr = torchaudio.load(sample_audio_path)
                if self.pre_process:
                    audio = torchaudio.functional.mu_law_encoding(audio, 16)
            
            audio = audio.squeeze()

            sample_audio = []
            for start, end in sample_audio_intervals:
                sample_audio.extend(audio[start:end])

            sample_audio = np.array(sample_audio)

            if np.isnan(skeletons).any():
                raise Exception("NaN value in sample!")

            ## LOAD Z
            z = np.load(self.z_path + str(idx) + "_z.npy")

            ## LABELS AND ONE-HOT 
            label = self.styles.index(sample_style)
            one_hot = np.zeros(len(self.styles))
            one_hot[label] = 1.0

            return [torch.Tensor(skeletons), torch.Tensor(sample_audio), torch.Tensor(z).view(512, -1, 1), torch.LongTensor([label]), torch.Tensor(one_hot)]

        except Exception as e:
            #print(e)
            #pdb.set_trace()
            return None

    def sort_key(self, string):
        return(int(string.split('.npy')[0]))

    def remove_style(self, metadata_dict, style_to_remove):
        n_samples = len(metadata_dict)
        filtered_dict = {}
        idx = 0

        for sample in range(n_samples):
            if style_to_remove != metadata_dict[sample]['style']:
                filtered_dict[idx] = metadata_dict[sample]
                idx = idx + 1

        return filtered_dict

    def make_z(self, idx, c=512, m=4):
        np.random.seed(idx)
        xs = np.linspace(0,1000,m) # Test input vector
        mxs = np.zeros(m) # Zero mean vector

        z = []
        for i in range(c):
            lsc = ((float(i)+1)/c)*(100*(1024/c))
            Kss = np.exp((-1*(xs[:,np.newaxis]-xs[:,np.newaxis ].T)**2)/(2*lsc**2)) # Covariance matrix
            fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
            z.append(fs)
        z = np.asarray(z)
        return z

def collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0:
        raise Exception("No sample on batch")
    return torch.utils.data.dataloader.default_collate(batch)
