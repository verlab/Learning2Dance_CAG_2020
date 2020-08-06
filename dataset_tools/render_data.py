from tqdm import tqdm
import numpy as np
import argparse
import librosa
import pickle
import cv2
import os

from tools import load_video, make_video

def parse_args():
    """ Parse input arguments """

    parser = argparse.ArgumentParser(description='Data augmentation.')

    parser.add_argument('--dataset_path', default="", type=str, help='Path to folder containing dataset.')
    parser.add_argument('--test_dataset', action="store_true", help='Create folder with 10 samples of dataset.')
    parser.add_argument('--metadata_path', default="", type=str, help="Path to metadata if test_dataset is True.")
    parser.add_argument('--render_video', default="", type=str, help="Specificy video to render.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    test_dataset = bool(args.test_dataset)
    render_jsons = bool(args.render_jsons)

    ## Test dataset
    if test_dataset:
        if args.metadata_path == "":
            raise("Please provide the path of metadata file.")

        with open(args.metadata_path, 'rb') as handle:
            metadata_dict = pickle.load(handle)

        ## Test output folder def
        path_idx = args.metadata_path.rfind('/')
        dataset_path = args.metadata_path[:path_idx]
        dataset_name = dataset_path.split('/')[-1]

        output_folder = args.metadata_path[:dataset_path.rfind('/')]
        output_folder = output_folder + '/' + 'test_' + dataset_name + '/'
        
        os.makedirs(output_folder, exist_ok=True)        

        ## ITERATE THROUGH DATA
        dataset_length = len(metadata_dict)
        rand_idxs = np.random.randint(low=0, high=dataset_length-1, size=10)

        for rand_idx in tqdm(rand_idxs, desc="Generating test videos..."):
            data_files = metadata_dict[rand_idx]['samples_frames']
            audio_intervals = metadata_dict[rand_idx]['samples_audio']
            audio_file = metadata_dict[rand_idx]['audio_path']

            ## LOAD VIDEO DATA
            video = load_video(data_files=data_files, data_path='')
            
            ## LOAD AUDIO FILE
            sample_rate = librosa.get_samplerate(audio_file)
            audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

            sample_audio = []
            for start, end in audio_intervals:
                sample_audio.extend(audio[start:end])

            sample_audio = np.array(sample_audio)

            ## PATH DEFS
            tmp = audio_file.split('/')
            style = tmp[-3]
            video_number = tmp[-2]

            video_name = output_folder + style + "_" + video_number + '.mp4'

            ## RENDER VIDEO
            video_path = dataset_path + '/' + style + '/' + video_number + '/' + video_number + '.mp4'
            cap=cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            make_video(video_name, video, fps)

            ## SAVE AUDIO FILE
            audio_name = output_folder + style + "_" + video_number + ".wav"
            librosa.output.write_wav(audio_name, sample_audio, sample_rate)

            ## EMBED AUDIO IN VIDEOs
            cmd = 'ffmpeg -loglevel error -i ' + video_name + ' -i ' + audio_name + ' -c:v copy -c:a aac ' + output_folder + 'output.mp4'
            os.system(cmd)

            ## RM FILES
            os.system('rm ' + audio_name)
            os.system('rm ' + video_name)
            os.system('mv ' + output_folder + 'output.mp4 ' + video_name)

    if args.render_video != "":
        video_dir = args.render_video 

        data_files = os.listdir(video_dir + '/data')
        data_files.sort(key=lambda x: int(x.split('.npy')[0]))

        video_path = video_dir + '/' + video_dir[video_dir.rfind('/')+1:] + '.mp4'

        print("Rendering video: " + video_dir[video_dir.rfind('/')+1:])

        video = load_video(data_files, video_dir + '/data/')

        video_name = video_dir + '/rendered.mp4'

        rendered_video = make_video(video_name, video, True, data_files)

if __name__ == '__main__':
    main()