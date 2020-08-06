from copy import deepcopy
from tqdm import tqdm
import argparse
import os

from tools import get_videos_path

def standardize_data(input_video, fps, audio_sr):
    output_video = deepcopy(input_video)
    index = output_video.rfind('I')
    output_video = output_video[:index] + 'temp.mp4'

    cmd = "ffmpeg -y -loglevel error -i " + input_video + " -r " + str(fps) + " -ac 1 -ar " + str(audio_sr) + " " + output_video
    os.system(cmd)
    os.system('rm ' + input_video)
    os.system('mv ' + output_video + ' ' + input_video) 

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Standardize dataset video fps and audio sample rate.')

    parser.add_argument('--dataset_path', default="", help='Path to dataset.')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Audio sample rate.')
    parser.add_argument('--fps', default=24, type=int, help='Video fps.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_path = get_videos_path(args.dataset_path)
    videos_path = [video_path + '/' + video_path.split('/')[-1] + '.mp4' for video_path in videos_path]

    for video in tqdm(videos_path, desc="Standardizing videos..."):
        standardize_data(video, args.fps, args.sample_rate)

if __name__ == '__main__':
    main()