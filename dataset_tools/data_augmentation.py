from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np
import argparse
import librosa
import os

from tools import get_videos_path, sort_npy, load_video

def flip_frame(skeleton, flip_map):
    for joint1, joint2 in flip_map:
        skeleton[[joint1, joint2]] = skeleton[[joint2, joint1]]

    return skeleton

def flip_video(video, flip_map):
    n_frames, _, _ = video.shape

    for frame in range(n_frames):
        video[frame, :, :] = flip_frame(video[frame, :, :], flip_map)

    return video

def add_noise(idx, video, joints_to_aug=list(range(25)), noise_x=True, noise_y=True, sigma=100, scale=80):
    np.random.seed(idx)

    t, n_joints, _ = video.shape

    joints_to_keep = [joint for joint in list(range(25)) if joint not in joints_to_aug]

    xs = np.linspace(0,1000,t) # Test input vector
    mxs = np.zeros(t) # Zero mean vector

    Kss = np.exp((-1*(xs[:,np.newaxis]-xs[:,np.newaxis ].T)**2)/(2*sigma**2)) # Covariance matrix

    zero_noise = np.zeros( (1, t) )

    ## NOISE X
    if noise_x:
        noise_x = []
        for i in range(n_joints):
            fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
            noise_x.append(fs)
        noise_x = np.asarray(noise_x).transpose(1, 0)

        for joint in joints_to_keep:
            noise_x[:, joint] = zero_noise

    else:
        noise_x = np.zeros( (t, n_joints) )

    ## NOISE Y
    if noise_y:
        noise_y = []
        for i in range(n_joints):
            fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
            noise_y.append(fs)
        noise_y = np.asarray(noise_y).transpose(1, 0)

        for joint in joints_to_keep:
            noise_y[:, joint] = zero_noise

    else:
        noise_y = np.zeros( (t, n_joints) )

    noise = np.array([noise_x, noise_y]).transpose(1, 2, 0)
    noise = noise/scale ## Scale noise
    
    ## ADDING NOISE TO DATA
    video = video + noise

    return video

def audio_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def audio_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def augment_audio(videos_dir, args, params):
    for video_dir in tqdm(sorted(videos_dir), desc='Processing videos...'):
        data_path = video_dir+'/'+video_dir.split('/')[-1]+'.wav'

        for i in range(len(params)):
            data, rate = librosa.load(data_path,16000)
            factor = i+1
            noise = audio_noise(data,0.05+(factor/100))
            pitch = audio_pitch(data,rate,factor)

            out_path = video_dir + '_' + str(i)
            librosa.output.write_wav(out_path+'/'+out_path.split('/')[-1]+'.wav',noise,rate)
            librosa.output.write_wav(out_path+'/'+out_path.split('/')[-1]+'_pitch.wav',pitch,rate)


def augment_data(videos_dir, args, params, i):

    (joints_to_aug, noise_x, noise_y, sigma, scale, flip, noise) = params

    flip_map = [
        (17, 18),
        (15, 16),
        (2, 5),
        (3, 6),
        (4, 7),
        (9, 12),
        (10, 13),
        (11, 14),
        (24, 21),
        (22, 19),
        (23, 20)
    ]

    idx = 0
    for video_dir in tqdm(videos_dir, desc='Processing videos...'):
        data_path = video_dir + '/data/'

        data_files = os.listdir(data_path)
        data_files = sorted(data_files, key=sort_npy)

        video = load_video(data_files, data_path)

        ## Augment data
        if noise:
            video = add_noise(idx, video, joints_to_aug, noise_x, noise_y, sigma, scale)
        if flip:
            video = flip_video(video, flip_map)

        output_path = video_dir + '_' + str(i) + '/data/'
        os.makedirs(output_path, exist_ok=True)

        for frame in range(video.shape[0]):
            np.save(output_path + str(frame) + '.npy', video[frame, :, :])
    
        ## Update seed to next video
        idx = idx + 1

def parse_args():
    """ Parse input arguments """

    parser = argparse.ArgumentParser(description='Data augmentation.')

    parser.add_argument('--dataset_path', default="", type=str, help='Path to folder containing dataset.')
    parser.add_argument('--pose',dest='pose', action='store_true',help='Augment pose data')
    parser.add_argument('--audio',dest='audio', action='store_true',help='Augment audio data')

    args = parser.parse_args()

    return args

## INIT PROGRAM
def main():
    args = parse_args()

    '''Params arguments
    (joints_to_aug, noise_x, noise_y, sigma, scale, flip, noise)'''

    params = [ # 40 # 30
        ([3, 4], True, True, 9.765625, 30, False, True),
        ([6, 7], True, True, 9.765625, 30, False, True),
        ([10, 11], True, True, 9.765625, 30, False, True),
        ([13, 14], True, True, 9.765625, 30, False, True),
        ([3, 4, 6, 7], True, True, 9.765625, 30, False, True),
        ([10, 11, 13, 14], True, True, 9.765625, 30, False, True),
        ([3 , 4, 10, 11], True, True, 9.765625, 30, False, True),
        ([6, 7, 13, 14], True, True, 9.765625, 30, False, True),
        ([3, 4, 13, 14], True, True, 9.765625, 30, False, True),
        ([6, 7, 10, 11], True, True, 9.765625, 30, False, True),
        ([3, 4], True, True, 9.765625, 40, False, True),
        ([6, 7], True, True, 9.765625, 40, False, True),
        ([10, 11], True, True, 9.765625, 40, False, True),
        ([13, 14], True, True, 9.765625, 40, False, True),
        ([3, 4, 6, 7], True, True, 9.765625, 40, False, True),
        ([10, 11, 13, 14], True, True, 9.765625, 40, False, True),
        ([3 , 4, 10, 11], True, True, 9.765625, 40, False, True),
        ([6, 7, 13, 14], True, True, 9.765625, 40, False, True),
        ([3, 4, 13, 14], True, True, 9.765625, 40, False, True),
        ([6, 7, 10, 11], True, True, 9.765625, 40, False, True)
    ]

    videos_dir = get_videos_path(args.dataset_path)
    videos_dir = [video_dir for video_dir in videos_dir if len(video_dir.split('/')[-1].split('_')) == 2]
    
    if args.pose:
        for i, param in enumerate(params):
            print("Processing dataset with params config " + str(i+1) + "/" + str(len(params)) )
            augment_data(videos_dir, args, param, i)
    
    if args.audio:
        print("Augmenting audio data...")
        augment_audio(videos_dir, args, params)

if __name__ == '__main__':
    main()