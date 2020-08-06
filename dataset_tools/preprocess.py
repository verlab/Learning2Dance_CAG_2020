from tqdm import tqdm
import numpy as np
import argparse
import os

from tools import sort_videos, get_videos_path, load_video_json

def treat_missing_joints(video):

    ## OPENPOSE DATASET
    normal_vectors_dict = {
        "head" : (0, 1),
        "body" : (1, 8),
        "legs" : (8, 1),
        "feet" : (8, 1)
    }

    joints_pairs = {
        0 : (17, 18),
        17 : 18,
        18 : 17,
        15 : 16,
        16 : 15,
        2 : 5,
        5 : 2,
        3 : 6,
        6 : 3,
        4 : 7,
        7 : 4,
        9 : 12,
        12 : 9,
        10 : 13,
        13 : 10,
        11 : 14,
        14 : 11,
        22 : 19,
        19 : 22,
        23 : 20,
        20 : 23,
        21 : 24,
        24 : 21
    }

    nvec_to_joints = {
        17 : "head",
        15 : "head",
        16 : "head",
        18 : "head",
        4 : "body",
        3 : "body",
        2 : "body",
        5 : "body",
        6 : "body",
        7 : "body",
        11 : "legs",
        10 : "legs",
        9 : "legs",
        12 : "legs",
        13 : "legs",
        14 : "legs",
        23 : "feet" ,
        22 : "feet" ,
        24 : "feet",
        21 : "feet",
        19 : "feet",
        20 : "feet"
    }

    for joint, sibling in joints_pairs.items():
        
        good_frames = np.where(video[:, joint, :].any(axis=1))[0]

        if len(good_frames) == 0:

            if joint == 0: ## Case when cant find joint 0.

                sibling0, sibling1 = sibling

                sibling0_good_frames = np.where(video[:, sibling0, :].any(axis=1))[0]
                sibling1_good_frames = np.where(video[:, sibling1, :].any(axis=1))[0]

                frame_idx = np.intersect1d(sibling0_good_frames, sibling1_good_frames)[0]

                video[frame_idx, 0, :] = (video[frame_idx, sibling0, :] + video[frame_idx, sibling1, :]) / 2

            else:
                sibling_good_frames = np.where(video[:, sibling, :].any(axis=1))[0]

                if len(sibling_good_frames) > 0: ## No treatment for missing joints with missing siblings.
                    
                    ## Point to mirror
                    sibling_frame = sibling_good_frames[0]
                    sibling_joint = video[sibling_frame, sibling, :]

                    skeleton_group = nvec_to_joints[joint]
                    p0_idx , p1_idx = normal_vectors_dict[skeleton_group]

                    ## Normal vec(N)
                    p0 = video[sibling_frame, p0_idx, :]
                    p1 = video[sibling_frame, p1_idx, :]

                    n_vec = p1 - p0
                    n_vec = n_vec/np.linalg.norm(n_vec) ## Norm 1 normal vector

                    ## Incidence vector(D)
                    d_vec = p0 - sibling_joint

                    ## Reflection vector(R) | R = D - 2(D.N)N
                    r_vec = d_vec - 2*np.dot(d_vec, n_vec)*n_vec

                    ## Reflected JOINT
                    reflected_joint = p0 + r_vec

                    ## Case when body is perpendicular to camera frame.
                    if np.allclose(reflected_joint, sibling_joint):
                        reflected_joint[0] = reflected_joint[0] + 1
                    
                    video[sibling_frame, joint, :] = reflected_joint

                else:
                    if joint == 15:

                        joint0, joint1 = (17, 0)

                        joint0_good_frames = np.where(video[:, joint0, :].any(axis=1))[0]
                        joint1_good_frames = np.where(video[:, joint1, :].any(axis=1))[0]

                        frame_idx = np.intersect1d(joint0_good_frames, joint1_good_frames)[0]

                        mean_horizontal = (video[frame_idx, joint0, 0] + video[frame_idx, joint1, 0])/2
                        half_dist = np.abs(mean_horizontal - video[frame_idx, joint0, 0])

                        video[frame_idx, joint, 0] = mean_horizontal
                        video[frame_idx, joint, 1] = video[frame_idx, joint0, 1] + half_dist

                    if joint == 16:

                        joint0, joint1 = (18, 0)

                        joint0_good_frames = np.where(video[:, joint0, :].any(axis=1))[0]
                        joint1_good_frames = np.where(video[:, joint1, :].any(axis=1))[0]

                        frame_idx = np.intersect1d(joint0_good_frames, joint1_good_frames)[0]

                        mean_horizontal = (video[frame_idx, joint0, 0] + video[frame_idx, joint1, 0])/2
                        half_dist = np.abs(mean_horizontal - video[frame_idx, joint0, 0])

                        video[frame_idx, joint, 0] = mean_horizontal
                        video[frame_idx, joint, 1] = video[frame_idx, joint0, 1] + half_dist

    return video

def normalize_kp(kp):
    kp = np.where(kp[:,:]!=0,kp[:,:],np.nan)
    x_min,y_min = np.nanmin(kp[:,0],axis=0),np.nanmin(kp[:,1],axis=0)
    x_max,y_max = np.nanmax(kp[:,0],axis=0),np.nanmax(kp[:,1],axis=0)
    kp_normalized = kp.copy()

    kp_normalized[:,0] = ( (kp[:,0]-((x_max+x_min)/2)) * (kp[:,0] != 0) )
    kp_normalized[:,1] = ( (kp[:,1]-((y_max+y_min)/2)) * (kp[:,1] != 0) )

    diag = np.linalg.norm(np.array([x_max,y_max])-np.array([x_min,y_min]))

    scale = 2/diag

    kp_normalized[:,0] = (kp_normalized[:,0] * scale)
    kp_normalized[:,1] = (kp_normalized[:,1] * scale)

    kp_normalized[:,0] = ((kp_normalized[:,0]+1)/2)
    kp_normalized[:,1] = ((kp_normalized[:,1]+1)/2)
    
    kp_normalized = np.nan_to_num(kp_normalized)
    return np.array(kp_normalized)

def motion_filter(video): 

    n_frames, n_joints, n_dim = video.shape

    problematic_joints = {
        8:1,
        2:1,
        5:1,
        0:1,
        3:2,
        4:3,
        6:5,
        7:6,
        9:8,
        12:8,
        10:9,
        13:12,
        11:10,
        14:13,
        15: 0,
        16: 0,
        17: 0,
        18: 0,
        19: 14,
        20: 14,
        21: 14,
        22: 11,
        23: 11,
        24: 11
    }

    for joint, parent in problematic_joints.items():
        problem_frames = np.where(~video[:, joint, :].any(axis=1))[0]

        if len(problem_frames) > 0:
            good_frames = np.where(video[:, joint, :].any(axis=1))[0]
            for frame in problem_frames:
                try:
                    if np.min(np.abs(good_frames - frame)) != (good_frames-frame)[np.argmin(np.abs(good_frames - frame))]: ## True pega o da esquerda
                        closest_frame = frame-np.min(np.abs(good_frames - frame))
                    else:
                        closest_frame = np.min(np.abs(good_frames - frame))+frame

                except Exception as e:
                    print(e)
                    pdb.set_trace()

                closest_parent = video[closest_frame, parent, :]
                actual_parent = video[frame, parent, :]

                parent_delta = actual_parent - closest_parent

                closest_joint = video[closest_frame, joint, :]

                video[frame, joint, :] = closest_joint + parent_delta

    return video

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Preprocess dataset.')

    parser.add_argument('--dataset_path', default="", help='Path to dataset.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    ## MOTION FILTER + SKELETON NORM (SAVE NUMPY)
    videos_dirs = get_videos_path(args.dataset_path)

    for video_dir in tqdm(videos_dirs, desc='Processing videos...'):
        ## Create folder for .npy files
        os.makedirs(video_dir + '/data/', exist_ok=True)

        ## Get jsons files
        jsons_path = video_dir +  '/openpose/json/' 
        jsons_files = os.listdir(jsons_path)
        jsons_files = sorted(jsons_files, key=sort_jsons)

        frames = load_video_json(jsons_files, jsons_path)
        frames = treat_missing_joints(frames)

        frames = np.array( [normalize_kp(frames[frame, :, :]) for frame in range(frames.shape[0])] )   
            
        frames = motion_filter(frames)

        n_samples, _, _ = frames.shape
        for sample in range(n_samples):
            np.save(video_dir + '/data/' + jsons_files[sample].split('.json')[0] + '.npy', frames[sample, :, :])

if __name__ == '__main__':
    main()