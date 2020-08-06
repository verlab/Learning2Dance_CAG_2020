from tqdm import tqdm
import numpy as np
import argparse
import math
import json
import cv2
import os

from tools import get_videos_path, read_json, write_json, draw

def sort_jsons(string): ## Used for small_dance_v2 pattern (CAUTION)
    return int(string.split('.')[0].split('_')[2])

def render_video(video_name, video_path):

    cap = cv2.VideoCapture(video_path + '/' + video_name)
    fps = int(math.ceil(cap.get(5))) #FPS
    width = int(cap.get(3))          #WIDTH
    height = int(cap.get(4))         #HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_debug = cv2.VideoWriter(video_path + '/debug.mp4', fourcc, fps, (640, 480) )

    i = 0
    ret = True
    while(ret): 
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.putText(frame, str(i), (10,50), font, 1,(0,0,255),2,cv2.LINE_AA) #Draw the text
            out_debug.write(frame)
        i = i + 1

    out_debug.release()
    cap.release()

def parse_args():
    parser = argparse.ArgumentParser(description="Render video with frame indexes.")

    parser.add_argument('--dataset_path', default= "", help='Path to root of dataset.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_dir = get_videos_path(args.dataset_path)
    videos_path = [video_dir + '/' + video_dir.split('/')[-1] + '.mp4' for video_dir in videos_dir]

    for idx, video_path in enumerate(tqdm(videos_path, desc="Rendering debug videos...")):
        video_name = video_path.split('/')[-1]

        filter_video(video_name, videos_dir[idx])

if __name__ == "__main__":
    main()