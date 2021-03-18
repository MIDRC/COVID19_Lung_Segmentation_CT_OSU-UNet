# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 07:29:51 2021

@author: cand07
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_rectangle(img, mask):
    " computes the non-zero borders, and draws a rectangle around the predicted lung area"
    " the area inside the rectangle can be sent to neural network to process" 
    
    " continue only if mask is not non-zero "
    if(np.count_nonzero(mask) !=0):
        vertical_sum = np.sum(mask, axis = 0)
        horizontal_sum = np.sum(mask, axis = 1) 
        
        indexes = np.nonzero(vertical_sum)[0]
        border_l = indexes[0]
        border_r = indexes[len(indexes)-1]
    
        indexes = np.nonzero(horizontal_sum)[0]
        border_up = indexes[0]
        border_down = indexes[len(indexes)-1]
    
        start_coord = ( border_l, border_up)
        end_coord = (border_r, border_down)
        color = (1, 1, 0) 
        thickness = 4

        maxval = np.max(img)
        minval = np.min(img)
        img = (img - minval)/(maxval-minval)
        img = cv2.rectangle(img, start_coord, end_coord, color, thickness) 

        # mask = mask.astype('int')
        mask = cv2.rectangle(mask, start_coord, end_coord, color, thickness) 

    return img, mask


def show_frames(img, pred, pr, img2):
    
    " show predicted results "
    plt.figure(1)
    plt.subplot(141)
    plt.axis('off')
    plt.title('input')
    plt.imshow(img, cmap=plt.cm.gray)
        
    plt.subplot(142)
    plt.axis('off')
    plt.title('Predicted Lung')
    plt.imshow(pred,cmap='jet')  
        
    plt.subplot(143)
    plt.axis('off')
    plt.title('Pred. th=0.5')
    plt.imshow(pr , cmap = plt.cm.gray)
    
    plt.subplot(144)
    plt.axis('off')
    plt.title('Segment')
    plt.imshow(img2 , cmap = plt.cm.gray)

    plt.show()
