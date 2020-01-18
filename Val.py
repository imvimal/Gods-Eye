# -- coding: utf-8 --
"""
Created on Fri Jul 20 20:06:33 2018

@author: poppinace
"""

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn.functional as F
#from torchvision import models
from sklearn.linear_model import LinearRegression
import os
import numpy as np
from PIL import Image
from time import time
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
from datetime import datetime
from IOtools import txt_write
from Network.SDCNet import SDCNet_VGG16_classify
from load_data_V2 import myDataset, ToTensor
from urllib.request import urlopen
import cv2
import ssl
from firebase import firebase

url = 'http://192.168.43.250:8080/photo.jpg'


def test_phase(opt):
    with torch.no_grad():
        label_indice = np.arange(0.5, 22+0.5 / 2, 0.5)
        add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
        label_indice = np.concatenate((add, label_indice))
        label_indice = torch.Tensor(label_indice)
        class_num = len(label_indice) + 1
        net = SDCNet_VGG16_classify(class_num, label_indice, psize=64,
                                    pstride=64, div_times=2, load_weights=True)
        mod_path = 'best_epoch.pth'
        mod_path = os.path.join('model/SHA', mod_path)
        all_state_dict = torch.load(mod_path, map_location=torch.device('cpu'))
        net.load_state_dict(all_state_dict['net_state_dict'])
        net.eval()
        root_dir=os.path.join(r'Test_Data','SH_partA_Density_map')
        #root_dir=os.path.join(r'Test_Data','SH_partB_Density_map')
        img_dir = os.path.join(root_dir, 'test', 'images')
        #img1_dir = os.path.join(root_dir, 'test', 'images')
        rgb_dir = os.path.join(root_dir, 'rgbstate.mat')
        #i=0
        while True:
            
            imgResp = urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            #img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(img_dir,'a.jpg'),img)
            
            img_dir = os.path.join(root_dir, 'test', 'images')
            #img_dir = os.path.join(img1_dir,'IMG_'+str(i+1)+'.jpg')
            testset = myDataset(img_dir, rgb_dir, transform=ToTensor(),
                                if_test=True, IF_loadmem=False)
            testloader = DataLoader(testset, batch_size=1,
                                    shuffle=False, num_workers=0)
            for j, data in enumerate(testloader):
                inputs = data['image']
                #inputs = Image.open('Test_Data/SH_partA_Density_map/test/images/x.jpg')
                 # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
                #h,w = inputs.size()[-2:]
                #ph,pw = (64-h%64),(64-w%64)
                # print(ph,pw)

                #if (ph!=64) or (pw!=64):
                    #tmp_pad = [pw//2,pw-pw//2,ph//2,ph-ph//2]
                    # print(tmp_pad)
                    #inputs = F.pad(inputs,tmp_pad)        
                #inputs = np.asarray(inputs).transpose(2, 0, 1)
                #inputs = torch.from_numpy(inputs)
                #print(type(inputs))
                inputs= inputs.type(torch.float32)
                features = net(inputs)
                div_res = net.resample(features)
                merge_res = net.parse_merge(div_res)
                outputs = merge_res['div'+str(net.args['div_times'])]
                del merge_res

                pre =  (outputs).sum()
                count=float(str(pre)[7:-1])
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                current_date= now.strftime('%d-%m-%Y')
                print("Count at ",current_time,"is ",count)
                fb = firebase.FirebaseApplication('https://crowd-analysis-c4a3b.firebaseio.com/',None)
                fb.put('/'+'location/'+opt['location']+'/'+current_date,current_time,count)
                #p_y.append(float(str(pre)[7:-1]))
                #p_x.append(float(str(datetime.datetime.now())[14:16]+'.'+str(datetime.datetime.now())[17:19]))
                break
            #i+=1
        
        
        





    im_num = len(testloader)
    '''
    print(p_x)
    print(p_y)
    p_x=np.asarray(p_x)
    p_y=np.asarray(p_y)
    p_x = p_x.reshape(-1, 1)  # values converts it into a numpy array
    p_y = p_y.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(p_x, p_y)  # perform linear regression
    Y_pred = linear_regressor.predict([[28.38]])  # make predictions
    print("Expected Crowd:"+str(Y_pred[0][0]))
    plt.plot(p_x,p_y,color='red')
    plt.show()
    '''
    return mae/(im_num), math.sqrt(rmse/(im_num)),  me/(im_num)







