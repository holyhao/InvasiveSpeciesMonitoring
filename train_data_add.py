#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:23:15 2017

@author: holy
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
import shutil
sample_submission = pd.read_csv("trainlabels.csv")
test_submit=pd.read_csv("submit_vgg16_400_64_60_0.5.csv")

for i in range(len(test_submit)):
    name =int(test_submit.iloc[i,0])
    lable=test_submit.iloc[i,1]
    namenew ='add'+str(name)
    if lable>0.9:        
        df2=pd.Series([namenew,'1'],index=['name','invasive'])
        sample_submission=sample_submission.append(df2,ignore_index=True)
        #shutil.copy("test/"+str(name)+'.jpg', 'train2/'+namenew+'.jpg')
    elif lable<0.1:
        df2=pd.Series([namenew,'0'],index=['name','invasive'])
        sample_submission=sample_submission.append(df2,ignore_index=True)
        #shutil.copy("test/"+str(name)+'.jpg', 'train2/'+namenew+'.jpg')
sample_submission.to_csv("newtrainlabels.csv", index=False)
