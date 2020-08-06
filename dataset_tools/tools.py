from tqdm import tqdm
import numpy as np
import json
import os

def sort_npy(string):
    return(int(string.split('.npy')[0]))

def sort_jsons(string):
    try:
        number = int(string.split('_')[2])
    except:
        number = int(string.split('.')[0])

    return number

def sort_openpose_jsons(string):
    return int(string.split('.')[0].split('_')[2])

def sort_videos(string):
    return(int(string.split('_')[1]))

def listdir_abs(path):
    return [path + element for element in os.listdir(path)]

def get_dirs(path):
    dirs_abs = filter(os.path.isdir, listdir_abs(path))
    dirs_relative = [dir.split('/')[-1] for dir in dirs_abs]
    dirs_relative = [dir for dir in dirs_relative if 'z_data' not in dir] ## Filter non style directories

    return dirs_relative

def get_videos_path(dataset_path):

    videos_path = []

    styles = get_dirs(dataset_path)

    for style in styles:
        style_path = dataset_path + style + '/'
        style_videos_path = get_dirs(style_path)
        
        videos_path.extend([style_path + video_path for video_path in style_videos_path])

    return videos_path

def load_video(data_files, data_path):

    n_frames = len(data_files)
    video = np.empty( (n_frames, 25, 2) )

    for idx, frame in enumerate(data_files):
        pose = np.load(data_path + frame)
        video[idx, :, :] = pose

    return video

def load_video_json(jsons, data_path):

    n_frames = len(jsons)
    video = np.empty( (n_frames, 25, 2) )

    problem_frames = []

    for idx, frame in enumerate(jsons):
        with open(data_path + frame) as f:
            try:
                pose = np.reshape(np.array(json.load(f)['people'][0]['pose_keypoints_2d']),(25,3))[:,0:2]
            except:
                problem_frames.append(int(frame.split('_')[2]))
                continue
            video[idx, :, :] = pose

    ## Remove empty frames loaded from openpose.
    video = np.delete(video, problem_frames, 0)

    return video

## FILTER OPENPOSE FUNCTIONS
def read_json(json_path, openpose_new=True):
    
    with open(json_path) as f:
        data = json.load(f)
    kps = []

    if openpose_new:
        for people in data['people']:
            kp = np.concatenate((np.array(people['pose_keypoints_2d']).reshape(-1, 3),np.array(people['face_keypoints_2d']).reshape(-1, 3),np.array(people['hand_left_keypoints_2d']).reshape(-1, 3),np.array(people['hand_right_keypoints_2d']).reshape(-1, 3)),axis=0)             
            kps.append(kp)
    else:
        for people in data['people']:
            kp = np.array(people['pose_keypoints']).reshape(-1, 3)
            kps.append(kp)

    return kps

def write_json(kps, json_path, lista, openpose_new=True):
    kps = np.array(kps)

    my_json = {'version':1.2,'people':[]}

    if openpose_new:
        for i in lista:
            keypoints = {'pose_keypoints_2d':kps[i][0:25].ravel().tolist(),'face_keypoints_2d':[],'hand_left_keypoints_2d':[],'hand_right_keypoints_2d':[]}
            my_json['people'].append(keypoints)
    else:
        for i in lista:
            keypoints = {'pose_keypoints':kps[i].ravel().tolist()}
            my_json['people'].append(keypoints)

    with open(json_path, 'w') as f:     
        json.dump(my_json,f)

def get_bbox(json_path, vis_thr=0.2):

    kps = read_json(json_path)
    # Pick the most confident detection.
    
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        center = (min_pt + max_pt) / 2.
        scale = 0.0

    else:
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height

    return scale, center

def draw(points,image):
    import cv2
    # following the pose give by https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/doc/media/keypoints_pose_25.png
    colors_rgb = {
        0 : (252, 146, 114), # nose
        1 : (215, 48, 39), #chest
        2 : (255, 102, 0), #R sholder
        3 : (252, 141, 89), #R cotovelo
        4 : (255,255,0), #R hand
        5 : (153, 204, 0), #L sholder
        6 : (102, 255, 51), #L cotovelo
        7 : (0, 255, 0), #L hand
        8 : (215, 48, 39), #bacia
        9 : (102, 255, 153), #R upper leg
        10 : (0, 255, 204), #R joelho
        11 : (102, 255, 255), #R foot
        12 : (51, 153, 255), #L upper leg
        13 : (0, 102, 255), #L joelho
        14 : (0, 0, 255), #L foot
        15 : (204, 51, 153), #R face
        16 : (153, 0, 204), #L face
        17 : (255, 51, 204), #R ear
        18 : (102, 0, 255), #L ear
        19 : (0, 0, 255), 
        20 : (0, 0, 255), 
        21 : (0, 0, 255), 
        22 : (102, 255, 255), 
        23 : (102, 255, 255), 
        24 : (102, 255, 255) 
    }

    colors_bgr = {}
    for joint,color in colors_rgb.items():
        colors_bgr.update({int(joint):(color[2],color[1],color[0])})
    points = np.array(points)

    p0,c0 = None,None
    #draw joints
    for joint,point in enumerate(points):
        if int(point[0]) != 0 and int(point[1]) != 0:
            cv2.circle(image,(int(point[0]),int(point[1])),10,colors_bgr[joint],-1)

    lines = {
        0:1,
        1:8,
        2:1,
        3:2,
        4:3,
        5:1,
        6:5,
        7:6,
        9:8,
        10:9,
        11:10,
        22:11,
        23:22,
        24:11,
        12:8,
        13:12,
        14:13,
        21:14,
        19:14,
        20:19,
        17:15,
        15:0,
        16:0,
        18:16
    }
    #conect joints
    for parent,joint in lines.items():
        if (int(points[parent][0]) != 0 and int(points[parent][1]) != 0) and (int(points[joint][0]) != 0 and int(points[joint][1]) != 0):
            cv2.line(image,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_bgr[parent],5,8,0)
    return image

def make_video(name,predictions,write_frame_tag,data_files):
    # following the pose give by https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/doc/media/keypoints_pose_25.png
    import cv2

    n = predictions.shape[0]

    h = 1000
    w = 1000

    out_video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))

    colors_rgb = {
        0 : (float(252/255), float(146/255), float(114/255)), # nose
        1 : (float(215/255), float(48/255), float(39/255)), #chest
        2 : (float(255/255), float(102/255), float(0/255)), #R sholder
        3 : (float(252/255), float(141/255), float(89/255)), #R cotovelo
        4 : (float(255/255), float(255/255), float(0/255)), #R hand
        5 : (float(153/255), float(204/255), float(0/255)), #L sholder
        6 : (float(102/255), float(255/255), float(51/255)), #L cotovelo
        7 : (float(0/255), float(255/255), float(0/255)), #L hand
        8 : (float(215/255), float(48/255), float(39/255)), #bacia
        9 : (float(102/255), float(255/255), float(153/255)), #R upper leg
        10 : (float(0/255), float(255/255), float(204/255)), #R joelho
        11 : (float(102/255), float(255/255), float(255/255)), #R foot
        12 : (float(51/255), float(153/255), float(255/255)), #L upper leg
        13 : (float(0/255), float(102/255), float(255/255)), #L joelho
        14 : (float(0/255), float(0/255), float(255/255)), #L foot
        15 : (float(204/255), float(51/255), float(153/255)), #R face
        16 : (float(153/255), float(0/255), float(204/255)), #L face
        17 : (float(255/255), float(51/255), float(204/255)), #R ear
        18 : (float(102/255), float(0/255), float(255/255)), #L ear
        19 : (float(0/255), float(0/255), float(255/255)), 
        20 : (float(0/255), float(0/255), float(255/255)), 
        21 : (float(0/255), float(0/255), float(255/255)), 
        22 : (float(102/255), float(255/255), float(255/255)), 
        23 : (float(102/255), float(255/255), float(255/255)), 
        24 : (float(102/255), float(255/255), float(255/255)) 
    }
    lines = {
        0:1,
        1:8,
        2:1,
        3:2,
        4:3,
        5:1,
        6:5,
        7:6,
        9:8,
        10:9,
        11:10,
        22:11,
        23:22,
        24:11,
        12:8,
        13:12,
        14:13,
        21:14,
        19:14,
        20:19,
        17:15,
        15:0,
        16:0,
        18:16
    }

    font = cv2.FONT_HERSHEY_SIMPLEX

    colors_bgr = {}
    images = []
    images_final = []
    images_final_white = []    
    try:
        for i, points in enumerate(predictions[0:n,:,:]): 
            for joint,color in colors_rgb.items():
                colors_bgr.update({int(joint):(color[2],color[1],color[0])})
            points = np.array(points)

            if write_frame_tag:
                image = np.ones((w,h,3),np.float32)
            else:
                image = np.zeros((w,h,3),np.float32)

            points = (points*2)-1
            points[:,0] = points[:,0]*(4/4)*350
            points[:,1] = points[:,1]*350
            points[:,0] = points[:,0] + 500
            points[:,1] = points[:,1] + 500

            #draw joints
            for joint,point in enumerate(points):
                if  (point[0] != 0) and (point[1] != 0):
                    if write_frame_tag:
                        cv2.circle(image,(int(point[0]),int(point[1])),5,colors_bgr[joint],-1)
                    else:
                        cv2.circle(image,(int(point[0]),int(point[1])),5,colors_rgb[joint],-1)
            #connect joints
            for parent,joint in lines.items():
                if ((points[parent][0] != 0) and (points[parent][1] != 0)) and ((points[joint][0] != 0) and (points[joint][1] != 0)):
                    if write_frame_tag:
                        cv2.line(image,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_bgr[parent],2,8,0)
                    else:
                        cv2.line(image,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_rgb[parent],2,8,0)

            if write_frame_tag:
                image = np.uint8(image) 
                cv2.putText(image, data_files[i], (10,50), font, 1,(0,0,255),2,cv2.LINE_AA) #Draw the text
                out_video.write((image*255))
            else:
                out_video.write((image*255).astype(np.uint8))

        out_video.release()

    except Exception as e:
        pdb.set_trace()
        f = open('exception.txt','a+')
        f.write(str(e)+'\n'+str(predictions)+'\n'+str(predictions.shape)+'\n--------')
        f.close()
        return np.ones((n,1000,1000,3))