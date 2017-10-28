#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:09:12 2017

@author: no1
"""
import tensorflow as tf  
import numpy  
import scipy.misc as misc
import os
import cv2
cwd = os.getcwd()

image=[]

for img_name in os.listdir(cwd):
    img_path = os.path.join(cwd , img_name)
    try:
        
        img = misc.imread(img_path)
        image.append(img)
    except:
        pass
    

