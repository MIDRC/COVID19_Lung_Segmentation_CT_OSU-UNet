# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:36:48 2021
            
@author: cand07
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
from tensorflow.keras.models import model_from_json
from utils import *






PATH = os.getcwd()
img_path = PATH + '/Data/'

" list all frames in the series "
frame_list = os.listdir(img_path)


" load lung segment model and weights... (model U-net) "
json_file = open('segment_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
        
" load weights into the model "
model.load_weights('segment_model.h5')
print("Loaded model from disk")


"input shape..."
seg_w = model.input.shape[1]
seg_h = model.input.shape[2]   


" read dicom sequences..."
for fname in frame_list:
    print(fname)   
    fname_with_path = img_path + fname
    
    " convert image to numpy array "
    itk_image = sitk.ReadImage(fname_with_path) 
    img = sitk.GetArrayFromImage(itk_image) 
    img = np.squeeze(img)
    img = img.astype('float32')
    "original size of the image"
    img_w, img_h = img.shape 
    
    " resize the image for prediction"    
    img_p = cv2.resize(img, dsize=(seg_w, seg_h))
    img_p = (img_p - np.min(img_p)) / (np.max(img_p) - np.min(img_p))
    x_test = np.expand_dims(img_p, axis = 2)
    
    " prediction..."
    x_test = np.expand_dims(x_test, axis = 0) 
    pred = model.predict(x_test)
    pred = pred.squeeze()
 
    " post-processing: convert probability to binary "
    pred = cv2.resize(pred, (img_w,img_h), interpolation = cv2.INTER_CUBIC)
    pr = pred > 0.5
    pr = pr.astype(int)
    img2, pr = draw_rectangle(img, pr) 
    
    " show predicted results "
    show_frames(img, pred, pr, img2)
    
        

    