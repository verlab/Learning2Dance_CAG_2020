from tools.stgc import *
from tools.graph import Graph
import torch
import torch.nn as nn


class cnn_1d_soudnet(torch.nn.Module):
    '''Class of the Model of a CNN-1D network inspired in the soundnet architecture'''
    def __init__(self, num_class):
        super(cnn_1d_soudnet, self).__init__()

        self.num_class = num_class
        self.audio_classification = audio_classification


        self.cnn1 = torch.nn.Conv1d(1,32,64,stride=2,padding=32)
        self.norm1 = torch.nn.BatchNorm1d(32)
        self.pool1 = torch.nn.MaxPool1d(4,stride=4)       

        self.cnn2 = torch.nn.Conv1d(32,64,32,stride=2,padding=16)
        self.norm2 = torch.nn.BatchNorm1d(64)
        self.pool2 = torch.nn.MaxPool1d(4,stride=4)       

        self.cnn3 = torch.nn.Conv1d(64,128,16,stride=2,padding=8)
        self.norm3 = torch.nn.BatchNorm1d(128)
        self.pool3 = torch.nn.MaxPool1d(4,stride=4)       

        self.cnn4= torch.nn.Conv1d(128,256,8,stride=2,padding=4)
        self.norm4 = torch.nn.BatchNorm1d(256)

        self.cnn5= torch.nn.Conv1d(256,1024,16,stride=12,padding=4)
        self.norm5 = torch.nn.BatchNorm1d(1024)

        self.fc1 = torch.nn.Linear(1024,self.num_class)

    def forward(self, inp):
        c = self.pool1(torch.nn.functional.leaky_relu(self.norm1(self.cnn1(inp))))
        c = self.pool2(torch.nn.functional.leaky_relu(self.norm2(self.cnn2(c))))
        c = self.pool3(torch.nn.functional.leaky_relu(self.norm3(self.cnn3(c))))
        c = torch.nn.functional.leaky_relu(self.norm4(self.cnn4(c)))
        c = torch.nn.functional.leaky_relu(self.norm5(self.cnn5(c)))

        c = torch.nn.functional.softmax(self.fc1(torch.nn.functional.adaptive_avg_pool1d(c.view(inp.shape[0],1,-1),1024)),dim=2).view(inp.shape[0],self.num_class)
        return c


class Generator(nn.Module):

    def __init__(self,device,num_class,dropout,train_phase=True,num_joints=25):
        super(Generator,self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        cols1 = [15,16,1,3,6,9,12,11,22,19,14]
        cols2 = [0,4,6]
        cols3 = [0]
        
        self.graph25 = Graph(25,[(0,1),(1,8),(2,1),(3,2),(4,3),(5,1),(6,5),(7,6),
                            (9,8),(10,9),(11,10),(22,11),(23,22),(24,11),
                            (12,8),(13,12),(14,13),(21,14),(19,14),(20,19),
                            (17,15),(15,0),(16,0),(18,16)],1)
        self.ca25 = torch.tensor(self.graph25.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a25 = torch.tensor(self.graph25.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _,l1 = self.graph25.getLowAjd(cols1)
        
        self.graph11 = Graph(11,l1,0)
        self.ca11 = torch.tensor(self.graph11.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a11 = torch.tensor(self.graph11.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _,l2 = self.graph11.getLowAjd(cols2)

        self.graph3 = Graph(3,l2,0)
        self.ca3 = torch.tensor(self.graph3.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a3 = torch.tensor(self.graph3.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _,l3 = self.graph3.getLowAjd(cols3)

        self.graph1 = Graph(1,l3,0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)
        ##############################
        #############END##############
        ##############################
        self.num_class = num_class
        self.num_joints = num_joints
        self.device = device
        self.train_phase = train_phase

        self.embed = nn.Embedding(self.num_class,512)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        self.act = self.lrelu


        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(32)
        self.norm5 = nn.BatchNorm2d(16)

        ########STGCN#######
        self.gcn0 = st_gcn(1024,512,(1,self.ca1.size(0)))
        self.gcn1 = st_gcn(512,256,(1,self.ca3.size(0)))
        self.gcn2 = st_gcn(256,128,(1,self.ca3.size(0)))
        self.gcn3 = st_gcn(128,64,(3,self.ca11.size(0)))
        self.gcn4 = st_gcn(64,32,(3,self.ca11.size(0)))
        self.gcn5 = st_gcn(32,16,(7,self.ca25.size(0)))
        self.gcn6 = st_gcn(16,2,(7,self.ca25.size(0)))
        #########END##########

        #######GRAPH-UPSAMPLING########
        self.ups1 = UpSampling(1,3,self.a3,1024)
        self.ups2 = UpSampling(3,11,self.a11,256)
        self.ups3 = UpSampling(11,25,self.a25,64)
        ###############END##############

        #######TEMPORAL-UPSAMPLING########
        self.upt1 = nn.ConvTranspose2d(256,256,(2,1),stride=(2,1))
        self.upt2 = nn.ConvTranspose2d(128,128,(2,1),stride=(2,1))
        self.upt3 = nn.ConvTranspose2d(64,64,(2,1),stride=(2,1))
        self.upt4 = nn.ConvTranspose2d(32,32,(2,1),stride=(2,1))
        ###############END##############
        
    def forward(self,y,z):
        #batch,channels,time,vertex
        ######CONDITIONING#########
        if self.train_phase:
            emb = self.embed(y).view(len(z),512,1,1).repeat(1,1,z.shape[2],1)
            inp = torch.cat((z,emb),1)
        else:
            ######TESTING CODE##########
            emb = self.embed(y).unsqueeze(2).repeat(1,1,4).permute(1,0,2).reshape(len(z),512,-1,1)
            inp = torch.cat((z[:,:,:emb.shape[2]],emb),1)
            ###########################
        ################################


        aux = self.lrelu(self.gcn0(inp,self.ca1))
        inp = aux

        aux = self.act(self.norm1(self.gcn1(self.ups1(inp),self.ca3)))
        aux = self.dropout(self.act(self.norm2(self.gcn2(self.upt1(aux),self.ca3))))
        aux = self.act(self.norm3(self.gcn3(self.ups2(self.upt2(aux)),self.ca11)))
        aux = self.dropout(self.act(self.norm4(self.gcn4(self.upt3(aux),self.ca11))))
        aux = self.act(self.norm5(self.gcn5(self.ups3(self.upt4(aux)),self.ca25)))
        
        aux = self.gcn6(aux,self.ca25)
        return aux
    
class Discriminator(nn.Module):

    def __init__(self,device,num_class,size_sample,num_joints=25):
        super(Discriminator,self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        cols1 = [15,16,1,3,6,9,12,11,22,19,14]
        cols2 = [0,4,6]
        cols3 = [0]
        
        self.graph25 = Graph(25,[(0,1),(1,8),(2,1),(3,2),(4,3),(5,1),(6,5),(7,6),
                            (9,8),(10,9),(11,10),(22,11),(23,22),(24,11),
                            (12,8),(13,12),(14,13),(21,14),(19,14),(20,19),
                            (17,15),(15,0),(16,0),(18,16)],1)
        self.ca25 = torch.tensor(self.graph25.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a25 = torch.tensor(self.graph25.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _,l1 = self.graph25.getLowAjd(cols1)
        
        self.graph11 = Graph(11,l1,0)
        self.ca11 = torch.tensor(self.graph11.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a11 = torch.tensor(self.graph11.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _,l2 = self.graph11.getLowAjd(cols2)

        self.graph3 = Graph(3,l2,0)
        self.ca3 = torch.tensor(self.graph3.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a3 = torch.tensor(self.graph3.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _,l3 = self.graph3.getLowAjd(cols3)

        self.graph1 = Graph(1,l3,0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)

        ##############################
        #############END##############
        ##############################

        self.size_sample = size_sample
        self.num_joints = num_joints
        self.device = device
        self.num_class = num_class

        self.embed = nn.Embedding(self.num_class,self.num_joints)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout()

        self.act = self.lrelu


        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)

        ########STGCN#######
        self.gcn0 = st_gcn(3,2,(7,self.ca25.size(0)))
        self.gcn1 = st_gcn(2,32,(7,self.ca25.size(0)))
        self.gcn2 = st_gcn(32,64,(3,self.ca11.size(0)))
        self.gcn3 = st_gcn(64,128,(3,self.ca11.size(0)))
        self.gcn4 = st_gcn(128,256,(1,self.ca3.size(0)))
        self.gcn5 = st_gcn(256,1,(1,self.ca1.size(0)))
        #########END##########

        #######GRAPH-DOWNSAMPLING########
        self.dws1 = DownSampling(25,11,self.a25,64)
        self.dws2 = DownSampling(11,3,self.a11,256)
        self.dws3 = DownSampling(3,1,self.a3,1)
        ###############END##############

        #######TEMPORAL-DOWNSAMPLING########
        self.dwt1 = nn.Conv2d(32,32,(int(self.size_sample/2)+1,1))
        self.dwt2 = nn.Conv2d(64,64,(int(self.size_sample/4)+1,1))
        self.dwt3 = nn.Conv2d(128,128,(int(self.size_sample/8)+1,1))
        self.dwt4 = nn.Conv2d(256,256,(int(self.size_sample/16)+1,1))
        self.dwt5 = nn.Conv2d(1,1,(int(self.size_sample/16),3))      
        ###############END##############


    def forward(self,x,y):

        #################CONDITIONING################
        emb = self.embed(y).view(len(x),1,1,self.num_joints).repeat(1,1,self.size_sample,1)
        aux = torch.cat((x,emb),1)
        inp = self.lrelu(self.gcn0(aux,self.ca25))
        ############################################

        # pdb.set_trace()
        aux = self.lrelu(self.dwt1(self.gcn1(inp,self.ca25)))
        aux = self.lrelu(self.norm1(self.dws1(self.dwt2(self.gcn2(aux,self.ca25)))))
        aux = self.lrelu(self.norm2(self.dwt3(self.gcn3(aux,self.ca11))))
        aux = self.lrelu(self.norm3(self.dws2(self.dwt4(self.gcn4(aux,self.ca11)))))
        aux = self.dwt5(self.gcn5(aux,self.ca3))
    
        return self.sigmoid(aux)