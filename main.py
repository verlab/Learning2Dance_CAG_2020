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
from tools.utils import *

from torch.utils.tensorboard import SummaryWriter
import scipy
from scipy.stats import multivariate_normal

import pdb

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
    data_aug = True, create_z=False, sample_rate=16000, keep_wav=True, styles_to_remove=[],pre_process=True)
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


                fake_pose = generator_network(labels,z)

                pred_fake = discriminator(fake_pose,labels)
               
                l1 = loss_l1(fake_pose,poses)
                l2 = loss_l2(fake_pose, poses)
                lg = bce_loss(torch.flatten(pred_fake),valid)
               
                lgen = args.lambda_l1*l1+args.lambda_discriminator*lg+args.lambda_l2*l2

                generator_optimizer_ft.zero_grad()
                lgen.backward(retain_graph=True)
                generator_optimizer_ft.step()

                fake_pose = fake_pose.detach()
                if flip:
                    discriminator_pred_fake = discriminator(poses,labels)
                    discriminator_pred_real = discriminator(fake_pose,labels)
                else:
                    discriminator_pred_fake = discriminator(fake_pose,labels)
                    discriminator_pred_real = discriminator(poses,labels)

                ld_real = bce_loss(torch.flatten(discriminator_pred_real),valid)
                ld_fake = bce_loss(torch.flatten(discriminator_pred_fake),fake)

                ld = (ld_real+ld_fake)*0.5
                discriminator_optimizer_ft.zero_grad()
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
            print('exece',e)
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


def init_weights(model):
    for param in model.parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

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
    model = Generator(device,args.num_class,args.dropout,False)
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

    parser.add_argument('--lambda_l2', dest='lambda_l2', type=int, default=0, help='Lambda L1')

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

if __name__ == '__main__':
    main()
