from tqdm import tqdm
import numpy as np
import argparse
import math
import json
import cv2
import os

from tools import get_videos_path, read_json, write_json, draw, sort_openpose_jsons

def find_nearest(array1, array2, value1, value2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    idx = np.sqrt((array1 - value1)**2 + (array2-value2)**2)

    return idx.argmin(),idx[idx.argmin()]

def get_skeleton(kps, json_file_name, aux_i, aux_j):
    '''
    Get the closest point to aux_i and aux_j
    if they are not -1, if they are -1 get the
    most confident values
    '''
    kps = np.array(kps)
    possibles_i = []
    possibles_j = []
    lista = list(range(len(kps)))
    for i in range(len(kps)):
        n = np.count_nonzero(kps[i][:,0][kps[i][:,0] > 0.1] )
        possibles_i.append((np.sum(kps[i][:,0][kps[i][:,0] > 0.1]))/n)
        n = np.count_nonzero(kps[i][:,0][kps[i][:,1] > 0.1] )
        possibles_j.append((np.sum(kps[i][:,1][kps[i][:,1] > 0.1]))/n)
    possibles_i = np.array(possibles_i)
    possibles_j = np.array(possibles_j)
    idx, distance = find_nearest(possibles_i,possibles_j,aux_i,aux_j)
    n = np.count_nonzero(kps[idx][:,0][kps[idx][:,0] > 0.1] )
    aux_i = (np.sum(kps[idx][:,0][kps[idx][:,0] > 0.1]))/n
    aux_j = (np.sum(kps[idx][:,1][kps[idx][:,1] > 0.1]))/n

    aux = lista[0]
    lista[0] = idx
    lista[idx] = aux
    write_json(kps,json_file_name,lista)
    
    return aux_i, aux_j, idx

def filter_video(video_name, json_files, video_path):
    os.makedirs(video_path + '/json_tmp/', exist_ok=True)

    cap = cv2.VideoCapture(video_path + '/' + video_name)
    fps = int(math.ceil(cap.get(5))) #FPS
    width = int(cap.get(3))   # WIDTH
    height = int(cap.get(4)) # HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_debug = cv2.VideoWriter(video_path + '/debug.mp4', fourcc, fps, (width, height) )

    i = 0
    point_x, point_y = width/2, height/2
    while(cap.isOpened() and i < len(json_files)):  
        ret, frame = cap.read()
        if ret:
            kps = read_json(video_path + '/openpose/json/' + json_files[i])
            if len(kps) != 0:
                #get closer skeleton of the previous one
                json_path = video_path + '/json_tmp/' + json_files[i].split('_')[2] + '.json'

                try:
                    point_x,point_y,idx = get_skeleton(kps, json_path, point_x, point_y)
                except Exception as e:
                    print(e)

                drawed_frame = draw(kps[idx],frame)
                drawed_frame = np.uint8(drawed_frame)
                cv2.putText(drawed_frame,json_files[i].split('/')[-1], (10,50), font, 1,(0,0,255),2,cv2.LINE_AA) #Draw the text
                out_debug.write(drawed_frame)
                
        i = i + 1

    ## RM NON FILTER DIR
    cmd = "rm -rf " + video_path + '/openpose/json/'
    os.system(cmd)

    ## MV FILTERED FILES TO OPENPOSE DIR
    cmd = "mv " + video_path + '/json_tmp ' + video_path + '/openpose/json'
    os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser(description="Filter openpose predictons, keeping only the best candidate.")

    parser.add_argument('--dataset_path', default= "", help='Path to root of dataset to extract audios.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    videos_dir = get_videos_path(args.dataset_path)
    videos_path = [video_dir + '/' + video_dir.split('/')[-1] + '.mp4' for video_dir in videos_dir]

    for idx, video_path in enumerate(tqdm(videos_path, desc="Filtering openpose predictions...")):
        video_name = video_path.split('/')[-1]
        json_files = os.listdir(videos_dir[idx] + '/openpose/json/')
        json_files = sorted(json_files, key=sort_openpose_jsons)

        filter_video(video_name, json_files, videos_dir[idx])

if __name__ == "__main__":
    main()