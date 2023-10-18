from math import floor, ceil
from tqdm import tqdm
import numpy as np
import argparse
import librosa
import pickle
import cv2
import os
import re

from tools import get_videos_path, sort_npy

def get_skeletons(npy_files, data_path, stride, sample_size=64):
    skeletons_dict = {}
    indices_dict = {}

    n_files = len(npy_files)

    samples_start = list( range(0, n_files+1, stride) )
    samples_start = [sample_start for sample_start in samples_start if (sample_start + sample_size) <= n_files ]

    n_samples = len(samples_start)

    for idx, sample in enumerate(range(n_samples)):
        start_index = samples_start[idx]
        sample_files = npy_files[start_index:(start_index+sample_size)]

        npys_full_path = [data_path + npy for npy in sample_files]
        skeletons_dict[sample] = npys_full_path
        
        indices = [int(npy.split('.npy')[0]) for npy in sample_files]
        indices_dict[sample] = indices

    return skeletons_dict, indices_dict

def map_seq_to_wav(frame_seq, video_fps, audio_sr):
    start_seq, end_seq = frame_seq

    video_start = start_seq/video_fps
    video_end = (end_seq+1)/video_fps ## Map until end of last frame.

    audio_start = audio_sr*video_start
    audio_end = audio_sr*video_end

    audio_start = floor(audio_start)
    audio_end = ceil(audio_end)

    ## Standardizing audio interval length
    n_audio_points = ceil((((end_seq-start_seq) + 1)/video_fps)*audio_sr)
    audio_interval = (audio_end - audio_start)

    audio_end = audio_end + (n_audio_points-audio_interval)

    return audio_start, audio_end

def find_sequences(video_frames):

    max_value = max(video_frames)
    min_value = min(video_frames)

    all_frames = list(range(min_value, max_value+1))

    ## Find missing frames
    missing_frames = np.array([int(frame not in video_frames) for frame in all_frames])
    ## Find missing frames indexes
    intervals = list(np.where(missing_frames == 1)[0] + min_value)
    ## Add initial and last frame to iterate over list
    intervals.insert(0, min_value-1) ## Get left value
    intervals.append(max_value+1)    ## Plus 1 for range func

    sequences = []

    for i in range(len(intervals)-1):
        start_index = intervals[i] + 1 ## Avoid missing indexes
        end_index = intervals[i+1]
        
        values = list(range(start_index, end_index))

        if len(values) > 0:
            sequences.append( (values[0], values[-1]) )

    return sequences

def map_video_to_audio(indices_dict, video_fps, audio_sr):

    audio_dict = {}

    for sample_idx, (_, sample_npys) in enumerate(indices_dict.items()):
        sample_sequences = find_sequences(sample_npys)
        audio_sequences = [map_seq_to_wav(seq_tuple, video_fps, audio_sr) for seq_tuple in sample_sequences]

        audio_dict[sample_idx] = audio_sequences

    return audio_dict

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Create metadata for l2d dataset.')

    parser.add_argument('--dataset_path', default="", help='Path to dataset.')
    parser.add_argument('--sample_size', default=64, type=int, help='Sample size.')
    parser.add_argument('--data_aug', action='store_true', help='Use or not augmented data.')
    parser.add_argument('--stride', default=32, type=int, help='Stride on data.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_dir = get_videos_path(args.dataset_path)

    if args.data_aug:
        wavs_paths =   [video_dir + '/' + video_dir.split('/')[-1] + '.wav' for video_dir in videos_dir]
        #wavs_paths =   [re.sub(r'(I_\d+)_\d+', r'\1', wav_path) for wav_path in wavs_paths] # Uncomment this line if you used audio data augmentation

        videos_paths = [video_dir + '/' + video_dir.split('/')[-1] + '.mp4' for video_dir in videos_dir]
        videos_paths = [re.sub(r'(I_\d+)_\d+', r'\1', video_path) for video_path in videos_paths]

    else:
        ## Remove data augmentation dirs
        videos_dir = [video_dir for video_dir in videos_dir if len(video_dir.split('/')[-1].split('_')) == 2]
        wavs_paths =   [video_dir + '/' + video_dir.split('/')[-1] + '.wav' for video_dir in videos_dir]
        videos_paths = [video_dir + '/' + video_dir.split('/')[-1] + '.mp4' for video_dir in videos_dir]

    videos_style = [video_dir.split('/')[-2] for video_dir in videos_dir]

    samples_dict = {}
    global_sample_idx = 0

    for idx, video_dir in enumerate(tqdm(videos_dir, desc='Processing videos...')):
        data_path = video_dir + '/data/'
        data_files = os.listdir(data_path)
        data_files = sorted(data_files, key=sort_npy)

        skeletons_dict, indices_dict = get_skeletons(data_files, data_path, args.stride, args.sample_size)

        ## Get audio sample rate
        wav_path = wavs_paths[idx]
        sample_rate = librosa.get_samplerate(wav_path)

        ## Get video fps
        video_path = videos_paths[idx]
        cap=cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        ## Map npy files to audio intervals/sequences
        audio_dict = map_video_to_audio(indices_dict, fps, sample_rate)

        ## Saving info
        for sample_idx, skeletons_sample in skeletons_dict.items():
            audio_sample = audio_dict[sample_idx]

            metadata = {}
            metadata['samples_frames'] = skeletons_sample
            metadata['samples_audio'] = audio_sample
            metadata['audio_path'] = wav_path
            metadata['style'] = videos_style[idx]
            metadata['video_id'] = idx

            samples_dict[global_sample_idx] = metadata

            global_sample_idx = global_sample_idx + 1

    ## Saving pickle file
    with open(args.dataset_path + 'metadata_' + str(args.sample_size) + '_' + str(args.stride) + '_' + str(int(args.data_aug)) + '.pickle', 'wb') as f:
        pickle.dump(samples_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()