from tqdm import tqdm
import argparse
import os

from tools import get_videos_path

def extract_audio(video_path):
    index = video_path.rfind('I')   
    full_video_name = video_path[index:]
    video_name = video_path[:index] + full_video_name.split('.mp4')[0]

    cmd = 'ffmpeg -loglevel error -i ' + video_path +' -vn -acodec copy ' + video_name + '.aac'
    os.system(cmd)

    cmd = 'ffmpeg -loglevel error -i ' + video_name + '.aac ' + video_name + '.wav'
    os.system(cmd)

    os.system('rm ' + video_name + '.aac')

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio data from videos in .wav format.")

    parser.add_argument('--dataset_path', default= "", help='Path to root of dataset to extract audios.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_path = get_videos_path(args.dataset_path)
    videos_path = [video_path + '/' + video_path.split('/')[-1] + '.mp4' for video_path in videos_path]

    for video in tqdm(videos_path, desc="Extracting audio from videos..."):
        extract_audio(video)

if __name__ == "__main__":
    main()