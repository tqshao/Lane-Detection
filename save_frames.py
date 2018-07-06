# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:23:12 2017

@author: Tianqu
"""

import cv2
vidcap = cv2.VideoCapture('000035.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1