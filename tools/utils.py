#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import torch
import numpy as np
import cv2
import tqdm
import argparse
import os
import time
import random
import json
import math

import sys
from data import *
from sigth2sound import *

from torch.utils.tensorboard import SummaryWriter
import scipy
from scipy.stats import multivariate_normal

import pdb

def normalize_kp(kp):
    kp = np.where(kp[:,:]!=0,kp[:,:],np.nan)
    x_min,y_min = np.nanmin(kp[:,0],axis=0),np.nanmin(kp[:,1],axis=0)
    x_max,y_max = np.nanmax(kp[:,0],axis=0),np.nanmax(kp[:,1],axis=0)

    kp_normalized = kp.copy()

    kp_normalized[:,0] = ( (kp[:,0]-((x_max+x_min)/2)) * (kp[:,0] != 0))
    kp_normalized[:,1] = ( (kp[:,1]-((y_max+y_min)/2)) * (kp[:,1] != 0) )

    diag = np.linalg.norm(np.array([x_max,y_max])-np.array([x_min,y_min]))

    scale = 2/diag

    kp_normalized[:,0] = (kp_normalized[:,0] * scale)
    kp_normalized[:,1] = (kp_normalized[:,1] * scale)

    kp_normalized[:,0] = ((kp_normalized[:,0]+1)/2)
    kp_normalized[:,1] = ((kp_normalized[:,1]+1)/2)

    return np.array(kp_normalized)

############DRAWING FUNCTIONS############
def draw_weights(name,model,writer,step):
    if name == 'generator_':
        writer.add_histogram(name+'UPS/1',model.ups1.w.weight.data.cpu(),step)
        writer.add_histogram(name+'UPS/2',model.ups2.w.weight.data.cpu(),step)
        writer.add_histogram(name+'UPS/3',model.ups3.w.weight.data.cpu(),step)

        writer.add_histogram(name+'GCN/0',model.gcn0.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/1',model.gcn1.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/2',model.gcn2.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/3',model.gcn3.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/4',model.gcn4.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/5',model.gcn5.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/6',model.gcn6.gcn.conv.weight.data.cpu(),step)

        writer.add_histogram(name+'UPT/1',model.upt1.weight.data.cpu(),step)
        writer.add_histogram(name+'UPT/2',model.upt2.weight.data.cpu(),step)
        writer.add_histogram(name+'UPT/3',model.upt3.weight.data.cpu(),step)
        writer.add_histogram(name+'UPT/4',model.upt4.weight.data.cpu(),step)
    else:
        writer.add_histogram(name+'DWS/1',model.dws1.w.weight.data.cpu(),step)
        writer.add_histogram(name+'DWS/2',model.dws2.w.weight.data.cpu(),step)
        writer.add_histogram(name+'DWS/3',model.dws3.w.weight.data.cpu(),step)

        writer.add_histogram(name+'GCN/0',model.gcn0.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/1',model.gcn1.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/2',model.gcn2.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/3',model.gcn3.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/4',model.gcn4.gcn.conv.weight.data.cpu(),step)
        writer.add_histogram(name+'GCN/5',model.gcn5.gcn.conv.weight.data.cpu(),step)

        writer.add_histogram(name+'DWT/1',model.dwt1.weight.data.cpu(),step)
        writer.add_histogram(name+'DWT/2',model.dwt2.weight.data.cpu(),step)
        writer.add_histogram(name+'DWT/3',model.dwt3.weight.data.cpu(),step)
        writer.add_histogram(name+'DWT/4',model.dwt4.weight.data.cpu(),step)
        writer.add_histogram(name+'DWT/5',model.dwt5.weight.data.cpu(),step)

    # params = []
    for i,(name,param) in enumerate(model.named_parameters()):
        # params.append(param.data.cpu())
        writer.add_histogram('All_parameters/'+str(i)+'_'+str(name),param.data.cpu(),step)

def render(predictions,n,normalized=3):
    """Image render to follow the trainig 
        using the tensorboard."""
    # following the pose give by https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/doc/media/keypoints_pose_25.png
    import cv2
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
    colors_bgr = {}
    images = []
    images_final = []
    images_final_white = []    
    try:
        aa = np.ones((25))
        b = [15,17,2,3,4,9,10,11,24,22,23]
        aa[b] = -1
        for points in predictions[0:n,:,:]: 
            for joint,color in colors_rgb.items():
                colors_bgr.update({int(joint):(color[2],color[1],color[0])})
            points = np.array(points)
            if normalized == 0:
                image = np.zeros((1080,1920,3))
            elif normalized == 1:
                image = np.zeros((1080,1920,3))                
            elif normalized == 2:

                image = np.zeros((360,640,3),np.float32)
                points[:,0] = points[:,0] + 320
                points[:,1] = points[:,1] + 180

            elif normalized == 3:

                image = np.zeros((360,640,3),np.float32)
                image_white = np.ones((360,640,3),np.float32)
                points = (points*2)-1
                points = points*90
                points[:,0] = points[:,0] + 320
                points[:,1] = points[:,1] + 180

            elif normalized == 4:
                image = np.zeros((1000,1000,3),np.float32)
                image_white = np.ones((1000,1000,3),np.float32)

            #draw joints
            for joint,point in enumerate(points):
                if  (point[0] != 0) and (point[1] != 0):
                    cv2.circle(image,(int(point[0]),int(point[1])),5,colors_rgb[joint],-1)
                    cv2.circle(image_white,(int(point[0]),int(point[1])),5,colors_rgb[joint],-1)
            #conect joints
            for parent,joint in lines.items():
                if ((points[parent][0] != 0) and (points[parent][1] != 0)) and ((points[joint][0] != 0) and (points[joint][1] != 0)):
                    cv2.line(image,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_rgb[parent],2,8,0)
                    cv2.line(image_white,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_rgb[parent],2,8,0)


            images_final.append(image)
            images_final_white.append(image_white)
        images_final = np.array(images_final)
        images_final_white = np.array(images_final_white)
        return images_final, images_final_white
    except Exception as e:
        pdb.set_trace()
        f = open('exception.txt','a+')
        f.write(str(e)+'\n'+str(predictions)+'\n'+str(predictions.shape)+'\n--------')
        f.close()
        return np.ones((n,360,480,3))

def draw_z(writer):
    z = torch.Tensor(make_z_vary(1024,64,4))
    for step,zl in enumerate(z):
        writer.add_histogram('Z/our',zl,step)
    z = torch.Tensor(make_z_vary(1024,720,45))
    for step,zl in enumerate(z):
        writer.add_histogram('Z/long',zl,step)



############LOSS FUNCTIONS############
def loss_cross_entropy(pred,target):
    return torch.nn.functional.cross_entropy(pred,target)

def loss_l2(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def loss_l1(pred, target):
    return torch.nn.functional.l1_loss(pred, target, size_average=None, reduce=None, reduction='mean')

def loss_generator(pred):
    eps = 1e-12
    return -torch.mean(torch.log(pred+eps))

def loss_discriminator(pred_real,pred_fake):
    eps = 1e-12
    return -torch.mean((torch.log(pred_real+eps) + torch.log(1-pred_fake+eps)))

############OUTPUT FUNCTIONS############
def write_jsons(out_path, predictions, n):
    if out_path.rfind('/') != len(out_path)-1:
        out_path = out_path + '/'

    os.makedirs(out_path + 'vid2vid/test_openpose/', exist_ok=True)

    ## DEFINITION OF VARS
    joints = list(range(25))

    json_dict = {}
    json_dict['version'] = 1.3
    json_dict['people'] = [
        {
            'hand_right_keypoints_2d':  list(np.zeros(63)),
            'face_keypoints_2d' :       list(np.zeros(210)),
            'hand_left_keypoints_3d' :  [],
            'pose_keypoints_3d' :       [],
            'pose_keypoints_2d' :       [],
            'face_keypoints_3d' :       [],
            'hand_left_keypoints_2d' :  list(np.zeros(63)),
            'hand_right_keypoints_3d' : []
        }
    ]
    ## ITERATION OVER FRAMES TO WRITE THEM TO .JSON FILES
    try:
        for frame_idx, frame in enumerate(predictions[0:n,:,:]):
            ## 'REVERSE' NORMALIZING
            frame = (frame*2)-1

            frame[:,0] = frame[:,0]*(4/4)*350
            frame[:,1] = frame[:,1]*350

            #frame = frame*350
            frame[:,0] = frame[:,0] + 500
            frame[:,1] = frame[:,1] + 500
            ## FILL KEYPOINTS DICT
            pose_keypoints_2d = []
            for joint in joints:
                pose_keypoints_2d.append(float(frame[joint, 0]))
                pose_keypoints_2d.append(float(frame[joint, 1]))
                pose_keypoints_2d.append(float(1.0))

            json_dict['people'][0]['pose_keypoints_2d'] = pose_keypoints_2d
            ## SAVE FILE
            with open(out_path + '/vid2vid/test_openpose/' + str(frame_idx).zfill(12) + '_keypoints.json', 'w') as fp:
                json.dump(json_dict, fp)

    except Exception as e:
        pdb.set_trace()
        print(e)

def make_video(name,predictions,n,normalized=3):
    """Image render to follow the trainig 
        using the tensorboard."""
    # following the pose give by https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/doc/media/keypoints_pose_25.png
    import cv2

    h = 1000
    w = 1000

    out_video = cv2.VideoWriter(name+'_black.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))
    out_video_white = cv2.VideoWriter(name+'_white.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))

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
    colors_bgr = {}
    images = []
    images_final = []
    images_final_white = []    
    try:

        # predictions = np.reshape(predictions,(n,25,2))
        for points in predictions[0:n,:,:]: 
            # points[:,0] = ((points[:,0]*aa)+1)/2
            # points[b,0] = points[b,0]-0.2
            for joint,color in colors_rgb.items():
                colors_bgr.update({int(joint):(color[2],color[1],color[0])})
            points = np.array(points)

            image = np.zeros((w,h,3),np.float32)
            image_white = np.ones((w,h,3),np.float32)
            points = (points*2)-1
            # points = points*121
            points[:,0] = points[:,0]*(4/4)*350
            points[:,1] = points[:,1]*350
            #points = points*350
            points[:,0] = points[:,0] + 500
            points[:,1] = points[:,1] + 500

            #draw joints
            for joint,point in enumerate(points):
                # if  (not np.isnan(point[0])) and (not np.isnan(point[1])):
                if  (point[0] != 0) and (point[1] != 0):
                    cv2.circle(image,(int(point[0]),int(point[1])),5,colors_rgb[joint],-1)
                    cv2.circle(image_white,(int(point[0]),int(point[1])),5,colors_rgb[joint],-1)
            #conect joints
            for parent,joint in lines.items():
                # if (not np.isnan(points[parent][0])) and (not np.isnan(points[parent][1])) or (not np.isnan(points[joint][0])) and (not np.isnan(points[joint][1])):
                if ((points[parent][0] != 0) and (points[parent][1] != 0)) and ((points[joint][0] != 0) and (points[joint][1] != 0)):
                    cv2.line(image,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_rgb[parent],2,8,0)
                    cv2.line(image_white,(int(points[parent][0]),int(points[parent][1])),(int(points[joint][0]),int(points[joint][1])),colors_rgb[parent],2,8,0)

            out_video.write((image*255).astype(np.uint8))
            out_video_white.write((image_white*255).astype(np.uint8))
        out_video.release()
        out_video_white.release()
    except Exception as e:
        pdb.set_trace()
        f = open('exception.txt','a+')
        f.write(str(e)+'\n'+str(predictions)+'\n'+str(predictions.shape)+'\n--------')
        f.close()
        return np.ones((n,1000,1000,3))

def do_splines(img_path):
    
    if os.path.isdir(img_path):
        processing_folder = True
        folder_name = img_path[:]
        
        output_dir_json = '/'.join(folder_name.split('/')[:-2]+['vid2vid_spline','test_openpose'])
        output_dir_image = '/'.join(folder_name.split('/')[:-2]+['vid2vid_spline','test_img'])
        os.makedirs(output_dir_json, exist_ok=True)
        os.makedirs(output_dir_image, exist_ok=True)
            

        # pdb.set_trace()
        images = [img_path+'/test_img/'+f for f in sorted(os.listdir(os.path.join(img_path + '/test_img/')))]
        #images = [im for im in images if not im.endswith('vis.jpg')]
        motion_sourceg = [img_path+'test_openpose/'+f for f in sorted(os.listdir(os.path.join(img_path + '/test_openpose/')))]

        # motion_sourceg = [ms for ms in motion_sourceg if not ms.endswith('.json')]

        if len(images) < 1:
            print ("subfolder test_image is empty")
            exit()
        if len(images) != len(motion_sourceg):
            print ("subfolder test_img and test_openpose does not have same size")
            pdb.set_trace()
            exit()
      
    else:
        print ("this path is not a folder")
        exit()
   
    #print (images)     

    my_shape = cv2.imread(images[0]).shape 

    window = 60
    gap_w = 20
      
    for i in range(int(math.floor(len(motion_sourceg)/(window)))):
        final = np.minimum(int((i +1)*window + gap_w),len(motion_sourceg)) 
        gap_final = final - (i +1)*window 
        gap_ini = gap_w         
        if i == 0:
            gap_ini = 0          
               

        motion_source = motion_sourceg[i*window - gap_ini:(i+1)*window + gap_final]           
        motion_reconstruction_spline(motion_source,gap_ini,gap_final)


    if int(math.floor(len(motion_sourceg)/(window)))*(window) < len(motion_sourceg):
        ini_tmp = int(math.floor(len(motion_sourceg)/(window)))*window
        ini = np.maximum(0,ini_tmp - gap_w) 
        gap_ini = ini_tmp - ini      
        
        motion_source = motion_sourceg[ini:len(motion_sourceg)]
        
        motion_reconstruction_spline(motion_source,gap_ini,0)

def mad(data, axis=None):
    from numpy import mean, absolute
    return mean(absolute(data - mean(data, axis)), axis)

def motion_smooth_spline(Jtr_motion, smoothing):
    from csaps import csaps

    # Set outliers bounds to get outlier joints that are far from median standard dev
    std_bounds = 2
    # let's accept a number max of 30% of outliers -- i.e. max of 8 joints
    rate_outliers = 0.3
    Njoints = 25
    k_mad = 1.48 # convertion constant of robust mad to a Gaussian std without outliers
    Nframes = len(Jtr_motion)

    # smooth joint 3D trajectories
    xjoints = [None] * Njoints
    yjoints = [None] * Njoints
    
    xjoints_sm = [None] * Njoints
    yjoints_sm = [None] * Njoints
    
    time = np.arange(Nframes)    
    
    error_pred = [None] * Njoints
    # first run per joint before outliers removal
    for ii in range(Njoints):
        xjoints[ii] = np.hstack([Jtr_motion[jj][ii,0] for jj in range(Nframes)])
        yjoints[ii] = np.hstack([Jtr_motion[jj][ii,1] for jj in range(Nframes)])
        
        poses = [xjoints[ii], yjoints[ii]]
        poses_sm = csaps(time, poses, time, smooth=smoothing)
        # make use the norm here
        error = np.sum(np.absolute(poses_sm - poses), axis = 0)
        error_pred[ii] = np.where(np.isnan(error), np.Inf, error)


    # voting scheme using robust median and MAD per joint
    outliers = [None] * Njoints
    outliers_cumul = np.zeros(len(time))
    for ii in range(Njoints):
        mediane = np.median(error_pred[ii])
        # std = k*mad -- k = 1.148
        made = mad(error_pred[ii])
        # test if values are afar of 2*std using robust mad
        outliers[ii] = (np.absolute(error_pred[ii] - mediane) > std_bounds*k_mad*made)
        #pdb.set_trace()
        outliers_cumul += outliers[ii].astype(int)
        
    xjoints = [None] * Njoints
    yjoints = [None] * Njoints
    
    ## let's accept a number max of 30% of outliers -- i.e. max of 8 joints
    max_outliers = Njoints*rate_outliers
    inlier_poses = (outliers_cumul < max_outliers)
    frame_inliers = time[inlier_poses]
    # spline with inliers per joint after outliers removal
    for ii in range(Njoints):
        xjoints[ii] = np.hstack([Jtr_motion[jj][ii,0] for jj in frame_inliers])
        yjoints[ii] = np.hstack([Jtr_motion[jj][ii,1] for jj in frame_inliers])        
        poses = [xjoints[ii], yjoints[ii]]
        poses_sm = csaps(frame_inliers, poses, time, smooth=smoothing)
        xjoints_sm[ii] = poses_sm[0, :]
        yjoints_sm[ii] = poses_sm[1, :]


    return [xjoints_sm, yjoints_sm,inlier_poses]

def motion_reconstruction_spline(motion_source,gap_ini,gap_final): 

    Jtr_motion = [None] * len(motion_source)

    for i in range(len(motion_source)):
        with open(motion_source[i]) as json_file:
            Jtr_motion[i] = np.reshape(np.array(json.load(json_file)['people'][0]['pose_keypoints_2d']),(25,3))[:,:2]
        #print (Jtr_motion[i].shape)       


    smoothing = 0.6  
    matrix = motion_smooth_spline(Jtr_motion, smoothing)



    for j in range(len(motion_source)):
        if (j >= gap_ini) and ((len(motion_source) - gap_final) > j):
            with open(motion_source[i]) as json_file:
                skeleton = json.load(json_file)   


            confi = np.ones(25)
            x_sm = [x[j] for x in matrix[0]]
            y_sm = [x[j] for x in matrix[1]]      

            openpose = np.stack([x_sm,y_sm,confi],axis=1)   
     
            skeleton['people'][0]['pose_keypoints_2d'] = openpose.ravel().tolist()

            json_name = '/'.join(motion_source[j].split('/')[:-3]+['vid2vid_spline','test_openpose',motion_source[j].split('/')[-1]])+ '_sm.json'
            jpg_name = '/'.join(motion_source[j].split('/')[:-3]+['vid2vid_spline','test_img',motion_source[j].split('/')[-1]])+ '_sm.jpg'
            with open(json_name, 'w') as f:
               json.dump(skeleton,f) 
            images_final, images_final_white = render(np.reshape(openpose,(1,25,3)),1,4)           
            cv2.imwrite(jpg_name,images_final[0]*255)
