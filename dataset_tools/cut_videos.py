from math import floor, ceil
from tqdm import tqdm
import argparse
import librosa
import cv2
import os
import re

from tools import get_videos_path

from moviepy.editor import VideoFileClip

def get_folder_id(string):
    return(int(string.split('_')[1]))

def extract_audio(video_path):
    index = video_path.rfind('I')   
    full_video_name = video_path[index:]
    video_name = video_path[:index] + full_video_name.split('.mp4')[0]

    cmd = 'ffmpeg -loglevel error -i ' + video_path +' -vn -acodec copy ' + video_name + '.aac'
    os.system(cmd)

    cmd = 'ffmpeg -loglevel error -i ' + video_name + '.aac ' + video_name + '.wav'
    os.system(cmd)

    os.system('rm ' + video_name + '.aac')
    
    audio_path = video_name + '.wav'

    return audio_path

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

def split_and_organize_data(video_intervals, video_path, style_dir):

    fps = int(VideoFileClip(video_path).fps)

    folders_ids = [get_folder_id(folder) for folder in os.listdir(style_dir)]
    max_folder_id = max(folders_ids)

    intervals = [list(range(start, end)) for (start, end) in video_intervals]

    cap = cv2.VideoCapture(video_path)
    fps = int(ceil(cap.get(5))) # FPS
    width = int(cap.get(3))     # WIDTH
    height = int(cap.get(4))    # HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    videos = []

    audio_path = extract_audio(video_path)

    sample_rate = librosa.get_samplerate(audio_path)

    audio, sr = librosa.load(audio_path, sr=sample_rate)

    videos_names = []
    audios_names = []

    for video_idx, video_interval in enumerate(video_intervals):
        out_folder_name = style_dir + 'I_' + str(max_folder_id + video_idx + 1)
        out_video_name = out_folder_name + '/I_' + str(max_folder_id + video_idx + 1) + '.mp4'
        out_audio_name = out_folder_name + '/I_' + str(max_folder_id + video_idx + 1) + '.mp3'

        videos_names.append(out_video_name)
        audios_names.append(out_audio_name)

        os.makedirs(out_folder_name, exist_ok=True)

        videos.append( cv2.VideoWriter(out_video_name, fourcc, fps, (width, height) ) )

        audio_start, audio_end = map_seq_to_wav(video_interval, fps, sr)

        librosa.output.write_wav(out_audio_name, audio[audio_start:audio_end], sr, False)
        
    i = 0
    ret = True
    while(ret):  
        ret, frame = cap.read()
        if ret:
            video_idx = [idx for idx, interval in enumerate(intervals) if i in interval]
            if video_idx != []:
                video_idx = video_idx[0]
                videos[video_idx].write(frame)
        i = i + 1

    for idx, video in enumerate(videos):
        video.release()

        tmp_name = videos_names[idx][:videos_names[idx].rfind('/')] + '/tmp.mp4'
        cmd = "ffmpeg -loglevel error -i " + videos_names[idx] + " -i " + audios_names[idx] + " -c:v copy -c:a aac " + tmp_name
        os.system(cmd)

        os.system("rm " + videos_names[idx])
        os.system("rm " + audios_names[idx])
        os.system("mv " + tmp_name + " " + videos_names[idx])
    
    cap.release()

    os.system("rm -rf " + video_path[:video_path.rfind('/')])

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Cut videos in specified intervals.')

    parser.add_argument('--dataset_path', default="", help='Path to dataset.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_dirs = get_videos_path(args.dataset_path)

    videos_paths = [video_dir + '/' + video_dir.split('/')[-1] + '.mp4' for video_dir in videos_dirs]
    txts_paths =   [video_dir + '/' + video_dir.split('/')[-1] + '.txt' for video_dir in videos_dirs]

    for idx, video_path in enumerate(tqdm(videos_paths, desc="Cutting videos...")):
        video_intervals = open(txts_paths[idx], 'r').read().splitlines()
        video_intervals = [video_interval.replace(' ', '') for video_interval in video_intervals if video_interval != '']
        video_intervals = [( int(interval.split(',')[0]) , int(interval.split(',')[1]) ) for interval in video_intervals]

        style_dir = videos_dirs[idx][:videos_dirs[idx].rfind('/')] + '/'

        split_and_organize_data(video_intervals, video_path, style_dir)

if __name__ == "__main__":
    main()