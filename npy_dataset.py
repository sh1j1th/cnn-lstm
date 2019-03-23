# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:07:19 2019

@author: shj_k
"""

import cv2
import numpy as np
import os
from keras.preprocessing import image

print(cv2.__version__)

parent_dir = 'raw_data/'
parent_dir = os.path.abspath(parent_dir)
frames_list = []
path = r'C:\Users\shj_k\Desktop\Project\dataset_npy'


num_vids = 3        #took only 5 vids due to limited cpu capacity
num_frames = 5     #took only 10 frames, decide as per your system specs
vid_count = 0
frame_count = 0

for act_class in os.listdir(parent_dir):
    vid_count = 0
    class_dir = os.path.join(parent_dir, act_class)
    vid_list = []
    for vid_file in os.listdir(class_dir):
        vid_count += 1
        if (vid_count<=num_vids):
            frames_list = []
            print ("IN: ",vid_file, vid_count)
            file = os.path.join(class_dir,vid_file)
            vid_cap = cv2. VideoCapture(file)
            success , frame = vid_cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame_count = 1
            while success:
#the below 2 lines of code helps remove empty frames
                blurred_image = cv2.GaussianBlur(frame, (7,7), 0)  
                canny2 = cv2.Canny(blurred_image, 50, 150)          #edge detection
                if (np.mean(canny2) != 0 and frame_count <= num_frames): # no edge means no character in frame
                    #print ("Frame read", frame_count)
                    #title = act_class + "_%d" %vid_count
                    #cv2.imwrite(os.path.join(path,"%d.jpg") %frame_count, gray_frame)
                    # loads RGB image as PIL.Image.Image type
                    #img = image.load_img(gray_frame, target_size=(120, 160))
                    # convert PIL.Image.Image type to 3D tensor with shape (120, 160, 3)
                    resized = cv2.resize(gray_frame, (128,128), interpolation = cv2.INTER_AREA)
                    x = image.img_to_array(resized) #use this line or append channel count to each frame
                    frames_list.append(x)
                    frame_count += 1
                success , frame = vid_cap.read()
                if success and frame_count <= num_frames:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            vid_list.append(frames_list)
        np.save(act_class,np.array(vid_list))