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

def collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0:
        raise Exception('No sample on batch')
    return torch.utils.data.dataloader.default_collate(batch)

def train_gcn(args,device):
    time_init = str(time.ctime(int(time.time()))).replace(" ","_")

    


    if args.summary is None:
        log_summary = '/tmp/summary/'
    else:
        log_summary = args.summary
    summary = SummaryWriter(log_dir=log_summary)
    gen_checkpoint_path = args.ckp_save_path+'_generator.pt'
    dis_checkpoint_path = args.ckp_save_path+'_discriminator.pt'
    try:
        os.makedirs(args.ckp_save_path,exist_ok=True)
    except Exception as e:
        pass

    dataset = pose_audio_dataset(args.dataset, sample_size=64, stride=32, \
    data_aug = 1, create_z=False, sample_rate=16000, keep_wav=True, styles_to_remove=[],pre_process=True)
    styles_dic = dataset.styles
    dataloaders = {
        'train' : torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate)
    }
    
    ##############################################
    #### NETWORKS INITIALIZATION #################
    ##############################################
    
    generator_network = Generator(device,args.num_class,args.dropout)
    generator_optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, generator_network.parameters()), lr=args.lr_g,betas=(0.5, 0.999))
    generator_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer_ft, step_size=30, gamma=0.1)

    generator_network.to(device)

    discriminator = Discriminator(device,args.num_class,args.size_sample)

    if args.adam:
        discriminator_optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_d,betas=(0.5, 0.999))
    else:
        discriminator_optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_d)
    discriminator_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer_ft, step_size=30, gamma=0.1)

    discriminator.to(device)

    generator_params,discriminator_params = sum(p.numel() for p in generator_network.parameters() if p.requires_grad ),sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    ######INIT WEIGHTS#######
    init_weights(generator_network)
    init_weights(discriminator)
    #########################
    
    if args.mse:
        bce_loss = torch.nn.MSELoss().to(device)
    else:
        bce_loss = torch.nn.BCELoss().to(device)
    

    ####################
    ##TRAINING LOOP#####
    ####################
    step = 0
    for epoch in range(args.epochs):
        # generator_network.train()
        # for batch_idx, (poses,labels) in enumerate(dataloaders['train']):
        try:
            for batch_idx, (poses, audio, z, labels, target) in enumerate(dataloaders['train']):

                generator_network.train()
                discriminator.train()


                valid = torch.Tensor(np.random.uniform(low=0.7, high=1.2, size=(len(poses)))).to(device)
                fake = torch.Tensor(np.random.uniform(low=0.0, high=0.3, size=(len(poses)))).to(device)
                # valid = torch.Tensor(np.random.uniform(low=1, high=1, size=(len(poses)))).to(device)
                # fake = torch.Tensor(np.random.uniform(low=0, high=0, size=(len(poses)))).to(device)

                #coin to cheat the discriminator
                flip = random.random() > args.flip

                poses = poses.permute(0,3,1,2).to(device)
                labels = labels.to(device)
                target = target.to(device)
                z = z.to(device)
                audio = audio.to(device)

                generator_optimizer_ft.zero_grad()

                fake_pose = generator_network(labels,z)

                pred_fake = discriminator(fake_pose,labels)
               
                l1 = loss_l1(fake_pose,poses)
                l2 = loss_l2(fake_pose, poses)
                lg = bce_loss(torch.flatten(pred_fake),valid)
               
                lgen = args.lambda_l1*l1+args.lambda_discriminator*lg+args.lambda_l2*l2


                lgen.backward(retain_graph=True)
                generator_optimizer_ft.step()

                discriminator_optimizer_ft.zero_grad()
                if flip:
                    discriminator_pred_fake = discriminator(poses,labels)
                    discriminator_pred_real = discriminator(fake_pose,labels)
                else:
                    discriminator_pred_fake = discriminator(fake_pose,labels)
                    discriminator_pred_real = discriminator(poses,labels)

                ld_real = bce_loss(torch.flatten(discriminator_pred_real),valid)
                ld_fake = bce_loss(torch.flatten(discriminator_pred_fake),fake)

                ld = (ld_real+ld_fake)*0.5

                ld.backward()
                discriminator_optimizer_ft.step()

                ######LOGS########
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t Loss: G: {:.5f}|D: {:.5f}'.format(
                        epoch, args.epochs ,batch_idx * len(poses), len(dataloaders['train'].dataset),
                        100. * batch_idx / len(dataloaders['train']),lgen.data.tolist(),ld.data.tolist()))
                step = step + 1
                summary.add_scalar('GCcGAN_Loss/Generator',lgen.data.tolist() , step)
                summary.add_scalar('GCcGAN_Loss/Discriminator',ld.data.tolist() , step)

                summary.add_scalar('GCcGAN_Loss_Generator/l1',l1.data.tolist() , step)
                summary.add_scalar('GCcGAN_Loss_Generator/l2',l2.data.tolist() , step)
                # summary.add_scalar('GCcGAN_Loss_Generator/l1_X',l1x.data.tolist() , step)
                # summary.add_scalar('GCcGAN_Loss_Generator/l1_Y',l1y.data.tolist() , step)
                summary.add_scalar('GCcGAN_Loss_Generator/lgan',lg.data.tolist() , step)
                # break


                ######ALL LOSS 1 GRAPH#####
                ####GENERATE TO MUCH FILES ON STORAGE FOR ALL SUMMAIRES##########
                # summary.add_scalars('GCcGAN_Loss_Generator/all',{'l1':l1.data.tolist(),'generator':lg.data.tolist()},step)
                # summary.add_scalars('GCcGAN_Loss/Both',{'generator':lgen.data.tolist(),'discriminator':ld.data.tolist()},step)
                # draw_weights('discriminator_',discriminator,summary,step)

        except Exception as e:
            # print('exece',e)
            #print(e)
            #pdb.set_trace()
            pass
            

        ##########DRAWWING WEIGHTS##########
        draw_weights('generator_',generator_network,summary,epoch)
        draw_weights('discriminator_',discriminator,summary,epoch)
        ####################################

        if epoch%100 == 99:

            ######making videos#####
            generator_network.eval()

            ##RENDER TRAIN IMAGES####
            if args.ft:
                draw_poses = generator_network(ft_vectors,z)
            else:
                draw_poses = generator_network(labels,z)
            output_array_gt = np.array(draw_poses.permute(0,2,3,1).cpu().data)
            for idx,pose in enumerate(output_array_gt):
                images,images_white = render(pose,args.size_sample)
                i = int(labels[idx,:])
                tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
                tensor_images_white = torch.Tensor(images_white).view(1,images_white.shape[0],images_white.shape[1],images_white.shape[2],images_white.shape[3]).permute(0,1,4,2,3)
                if i < 3:
                    summary.add_video('gif_train/black_train_'+str(styles_dic[i])+'_'+str(idx%10),tensor_images,epoch,15)
                    summary.add_video('gif_train/white_train_'+str(styles_dic[i])+'_'+str(idx%10),tensor_images_white,epoch,15)
                    # summary.add_images('images/white_train_images'+str(i%3),tensor_images_white[0],epoch)
                    # summary.add_images('images/black_train_images'+str(i%10),tensor_images[0],epoch)
                else:
                    break
            ##RENDER TEST IMAGES####
            # z_train = np.random.randint(0,len(dataset),10)
            # z_test = np.random.randint(len(dataset),2*len(dataset),len(z))
            # tensor_z = []
            # for idx in z_train:
            #     tensor_z.append(torch.Tensor(make_z_vary(idx,1024,64,4)).view(1024,-1,1))
            # for idx in z_test:
            #     tensor_z.append(torch.Tensor(make_z_vary(idx,512,64,4)).view(512,-1,1))
            # tensor_z = torch.stack(tensor_z).to(device)
            # labels = torch.LongTensor([0,1,2]).to(device)

            # if args.ft:
            #     draw_poses = generator_network(ft_vectors,tensor_z)
            # else:
            #     draw_poses = generator_network(labels,tensor_z)
            # output_array = np.array(draw_poses.permute(0,2,3,1).cpu().data)

            # for idx,pose in enumerate(output_array):
            #     if idx < 3:
            #         images,images_white = render(pose,args.size_sample)
            #         tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
            #         tensor_images_white = torch.Tensor(images_white).view(1,images_white.shape[0],images_white.shape[1],images_white.shape[2],images_white.shape[3]).permute(0,1,4,2,3) 

            #         i = int(labels[idx])
            #         # if i < 10:
            #         #     summary.add_video('gif/black_train'+str(i%10),tensor_images,epoch,15)
            #         #     summary.add_video('gif/white_train'+str(i%10),tensor_images_white,epoch,15)
            #         #     summary.add_images('images/white_train_images'+str(i%3),tensor_images_white[0],epoch)
            #         #     summary.add_images('images/black_train_images'+str(i%10),tensor_images[0],epoch)
            #         # else: 
            #         summary.add_video('gif_test/black_test_'+str(styles_dic[i])+'_'+str(idx%10),tensor_images,epoch,15)
            #         summary.add_video('gif_test/white_test_'+str(styles_dic[i])+'_'+str(idx%10),tensor_images_white,epoch,15)
            #         # summary.add_images('images/white_test_images'+str(i%3),tensor_images_white[0],epoch)
            #         # summary.add_images('images/black_test_images'+str(i%10),tensor_images[0],epoch)
            #     else:
            #         break


            # draw_poses = generator_network(None,z_mean)
            # output_array = np.array(draw_poses.permute(0,2,3,1).cpu().data)
            # gt_array = np.array(mean_kps.permute(0,2,3,1).cpu().data)
            # # # pdb.set_trace()
            # # for i,pose in enumerate(output_array):
            # #     images = render(pose,args.size_sample)
            # #     tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
            # #     if i < 1:
            # #         # summary.add_video('Prediction/train_'+str(i%5),tensor_images,epoch,15)
            # #         summary.add_video('Overfit/predict'+str(i%5),tensor_images,epoch,15)
            # #     else:
            # #         break
            # #         # summary.add_video('Prediction/test_'+str(i%5),tensor_images,epoch,15)
            # #         # summary.add_video('Overfit/test_'+str(i%5),tensor_images,epoch,15)
            # for i,pose in enumerate(gt_array):
            #     images,images_white = render(pose,args.size_sample)
            #     tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
            #     tensor_images_white = torch.Tensor(images_white).view(1,images_white.shape[0],images_white.shape[1],images_white.shape[2],images_white.shape[3]).permute(0,1,4,2,3)
            #     if i < 1:
            #         # summary.add_video('Prediction/train_'+str(i%5),tensor_images,epoch,15)
            #         summary.add_video('Overfit/gt'+str(i%5),tensor_images,epoch,15)
            #         summary.add_video('Overfit/white_gt'+str(i%5),tensor_images_white,epoch,15)
            #     else:
            #         break

            ###render recive array of shape n,25,2

            # poses_gt = np.array(poses.cpu().data)
            # for i in range(args.num_class):
            #     label = torch.LongTensor([i]).to(device)
            #     z1 = torch.Tensor(make_z(1,args.size_sample)).view(len(label),1,args.size_sample,25).to(device)
            #     fake_poses = generator_network(label,torch.cat((z1,mean_kp.repeat(len(z1),1,1,1)),1))    
            #     outputs_array = np.array(fake_poses.cpu().data)
            #     images = render(outputs_array[0,:,:,:],args.size_sample)
            #     # tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
            #     tensor_images = torch.Tensor(images).view(1,images.shape[0],images.shape[1],images.shape[2],images.shape[3]).permute(0,1,4,2,3)
            #     summary.add_video('Prediction/'+str(styles_dic[i]),tensor_images,epoch,15)
    torch.save(generator_network.state_dict(),gen_checkpoint_path)
    torch.save(discriminator.state_dict(), dis_checkpoint_path)
    return

def draw_z(writer):
    z = torch.Tensor(make_z_vary(1024,64,4))
    for step,zl in enumerate(z):
        writer.add_histogram('Z/our',zl,step)
    z = torch.Tensor(make_z_vary(1024,720,45))
    for step,zl in enumerate(z):
        writer.add_histogram('Z/long',zl,step)

def init_weights(model):
    for param in model.parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

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

def get_audio_torch(input_path):
    import torchaudio

    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] == 2:
        audio = audio.sum(axis=0)/2
        audio = audio.reshape(1,-1)
    audio = torchaudio.transforms.Resample(sr,16000)(audio)
    audio = torchaudio.functional.mu_law_encoding(audio,16)

    return audio.float()

def test(args,device):
     #actual 0->ballet;1->country;2->michael 
    #old 0->ballet;1->michael;2->country
    styles_dic = {0:0,1:2,2:1}


    audio_model = cnn_1d_soudnet(3)
    audio_model.load_state_dict(torch.load(args.a_ckp_path))
    audio_model.to(device)
    model = Generator(device,args.num_class,args.size_sample,args.act_layer_g,args.dropout,args.last_layer,args.type,args.ft,False)
    model.load_state_dict(torch.load(args.ckp_path))
    model.to(device)

    audio_model.eval()
    model.eval()
    
    audio = get_audio_torch(args.input)
    video_size = int((audio.shape[1]/16000)*args.fps)

    z = torch.Tensor(make_z_vary(None, 512, args.size_video, int(video_size/16))).view(1,512,-1,1).to(device)

    label = audio_model(audio[:,:int(int(audio.shape[1]/int(z.shape[2]/4))*int(z.shape[2]/4))].to(device).view(int(z.shape[2]/4),1,-1))

    label = label.argmax(1).cpu().data.to(device)
    draw_poses = model(label,z)

    notorch_pose = draw_poses[0].permute(1, 2, 0).cpu().data.numpy()
    try:
        os.mkdir(args.out_video)
    except:
        pass
    
    label_0 = label.cpu().data.tolist()[0]
    if label_0 == 0:
        video_name = '/ballet'
    elif label_0 == 1:
        video_name = '/michael'
    elif label_0 == 2:
        video_name = '/salsa'

    os.makedirs(args.out_video + video_name + '/vid2vid/test_img/', exist_ok=True)
    make_video(args.out_video + video_name + '/' + video_name, notorch_pose,video_size)

    f = open(args.out_video + video_name +'/labels.txt', "w")
    for l in label.cpu().data.tolist():
        if l == 0:
            f.write('ballet\t')        
        elif l == 1:
            f.write('michael\t')
        elif l == 2:
            f.write('salsa\t')
    f.close()

    cmd = "ffmpeg -loglevel error -i '" + args.out_video + video_name + '/' + video_name + "_black.mp4' '" + \
    args.out_video + video_name + '/vid2vid/test_img/' + video_name + "_%04d.jpg'"  
    os.system(cmd)

    write_jsons(args.out_video + video_name, notorch_pose, draw_poses.shape[2])

    if args.splines:
        do_splines(args.out_video + video_name + '/vid2vid/')
    return

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.phase in ['train_cgan','train']:
        train_gcn(args,device)
    else:
        test(args,device)
    return

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Sound2Sigth code for training and test.')

    parser.add_argument('-p', '--phase', dest='phase', default='train',help='demo or extract feature. e.g., [train OR test].')

    parser.add_argument('-d', '--dataset', dest='dataset', help='path to .csv file with the dataset info.')
    
    parser.add_argument('--ckp_save_path', dest='ckp_save_path', help='path to save checkpoints of the model')

    parser.add_argument('--summary_dir', dest='summary', help='path to save summaries of training')
    
    parser.add_argument('-e', '--epochs',type=int, dest='epochs', help='number of epochs', default='500')

    parser.add_argument('-b', '--batch_size',type=int, dest='bs', help='batch size', default='8')

    parser.add_argument('--lr_g', type=float, dest='lr_g', help='learning_rate', default='0.002')

    parser.add_argument('--lr_d', type=float, dest='lr_d', help='learning_rate', default='0.0002')

    parser.add_argument('-c', '--n_class', type=int, dest='num_class', help='number of class', default='3')

    parser.add_argument('-s', '--size_sample', type=int, dest='size_sample', help='number of frames to predict', default='64')

    parser.add_argument('--workers',dest='workers', type=int, default=12, help="number of cpu threads to use during batch generation")

    parser.add_argument('--l1',dest='l1',action='store_true',help='Use L1 norm in training')
    
    parser.add_argument('--mse',dest='mse',action='store_true',help='Use MSE instead of BCE in training')

    parser.add_argument('--adam',dest='adam',action='store_true',help='Use ADAM instead of SGD as discriminator optimizer')

    parser.add_argument('--flip', type=float, dest='flip', help='trick the discriminator fliping values', default='1')

    parser.add_argument('-k', '--cpk_path', dest='ckp_path', help='path to load the checkpoints of the model')

    parser.add_argument('-o', '--out_video', dest='out_video', help='path to save videos *.mp4')

    parser.add_argument('-n', '--n_videos', dest='n_videos', type=int, default=1, help='number of video to be generated')

    parser.add_argument('--size_video', dest='size_video', type=int, default=64, help='size of the video to be generated')

    parser.add_argument('--lambda_l1', dest='lambda_l1', type=int, default=100, help='Lambda L1')

    parser.add_argument('--lambda_discriminator', dest='lambda_discriminator', type=int, default=1, help='Lambda Discriminator')

    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='Use Dropout in the Generator')

    parser.add_argument('--audio_ckp', dest='a_ckp_path', help='path to checkpoints of the audio classifier model')

    parser.add_argument('-i', '--input', dest='input', help='input *.wav file with path sound to test')

    parser.add_argument('--fps', dest='fps', type=int, default=15, help='FPS of the final generate video.')

    parser.add_argument('--splines',dest='splines', default=True,action='store_true',help='Generate the output smooth by cubic splines')

    args = parser.parse_args()

    if args.phase != 'train' and args.phase != 'test':
        parser.error("[--phase] only works in train and test phases")
        sys.exit()

    if args.phase == 'train' and args.dataset is None:
        parser.error("[--phase] train requires [--dataset]")
        sys.exit()

    if args.phase == 'train' and args.ckp_save_path is None:
        parser.error("[--phase] train requires [--ckp_save_path]")
        sys.exit()
    
    if args.phase == 'test' and args.ckp_path is None:
        parser.error("[--phase] test requires [--cpk_path]\nAlso you probably would like to use [--n_videos] and [--size_video]")
        sys.exit()

    if args.phase == 'test' and args.a_ckp_path is None:
        parser.error("[--phase] test requires [--audio_ckp]\nAlso you probably would like to use [--n_videos] and [--size_video]")
        sys.exit()

    if args.phase == 'test' and args.out_video is None:
        parser.error("[--phase] test requires [--out_video]\nAlso you probably would like to use [--n_videos] and [--size_video]")
        sys.exit()

    if args.phase == 'test' and args.input is None:
        parser.error("[--phase] test requires [--out_video]\nAlso you probably would like to use [--n_videos] and [--size_video]")
        sys.exit()

    return args

def make_z_vary(idx,c,t,m):
    if idx is not None:
        np.random.seed(idx)
    else:
        np.random.seed()

    xs = np. linspace (0,1000,m) # Test input vector
    mxs = np.zeros(m) # Zero mean vector

    z = []
    for i in range(c):
        # lsc = ((float(i)+1)/1024)*100
        lsc = ((float(i)+1)/c)*(100*(1024/c))
        Kss = np.exp((-1*(xs[:,np.newaxis]-xs[:,np.newaxis ].T)**2)/(2*lsc**2)) # Covariance matrix
        fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
        z.append(fs)
    z = np.asarray(z)
    return z

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

if __name__ == '__main__':
    main()
