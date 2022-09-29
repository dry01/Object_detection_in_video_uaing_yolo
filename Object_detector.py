# object detection on video file


# packages import
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
#import matplotlib.pyplot as plt
# %matplotlib inline

import torch
from torch import nn
from torchvision import transforms

import os
import sys

# path of input video
fn = sys.argv[1]
if os.path.exists(fn):
    print(os.path.basename(fn))
    # file exists


# function to detect object and save output video
def object_detect(img, width, height):
     yoloModel = torch.hub.load("ultralytics/yolov5", "yolov5s")# load yolo model
     results = yoloModel(img, size=300)
     
     output_imgs = []
     for i in range(len(img)):
         imageIndex= results.pandas().xyxy[i]
         image = img[i]
         for j in  range(len(imageIndex)):
             startX = int(imageIndex["xmin"][j])
             startY = int(imageIndex["ymin"][j])
             endX = int(imageIndex["xmax"][j])
             endY = int(imageIndex["ymax"][j])
             

             y = startY - 10 if startY - 10 > 10 else startY + 10
             if float(imageIndex["confidence"][j]) >= 0.8 :

                    cv2.rectangle(image,
	                  (startX, startY), (endX, endY),(0, 255, 0), 2)  
                    cv2.putText(image, imageIndex["name"][j] + " {:.2f}".format(float(imageIndex["confidence"][j])),
	                  (startX, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
             else: 
                    cv2.rectangle(image,
	                  (startX, startY), (endX, endY),(0,0, 255), 2)
                    cv2.putText(image, imageIndex["name"][j] + " {:.2f}".format(float(imageIndex["confidence"][j])),
	                  (startX, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)


             
         #dim1 = (width, height)
         # displaying image with bounding box
         #cv2.imshow('face_detect', image)
         # loop will be broken when 'q' is pressed on the keyboard
         #if cv2.waitKey(10) & 0xFF == ord('q'):
         #    break
         final_img = cv2.resize(image,(width, height), interpolation = cv2.INTER_AREA)    
         output_imgs.append(final_img)
         
     return output_imgs

batch_size = 16

cap = cv2.VideoCapture(fn) # read video file
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # input frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # input frame height



fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("output_video.avi", fourcc, 20, (width, height)) # saving video output

out_img = [] # to generate output 
exit_flag = True
while exit_flag:
    batch_inputs = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if ret:
            dim = (300, 300)
            resized = cv2.resize(frame[:,:,::-1], dim, interpolation = cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch_inputs.append(resized)
        else:
            exit_flag = False
            break

    outputs = object_detect(img=batch_inputs, width=width, height=height)
    if outputs is not None:
        for output in outputs:
            out.write(output)
            out_img.append(output)
    else:
        exit_flag = False

# output
for i in range(len(out_img)):
     cv2.imshow('object_detect',out_img[i] )
     if cv2.waitKey(10) & 0xFF == ord('q'):
             break