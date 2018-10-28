######################################################################
#
#  ANPR Demo for UP Embedded Vision Starter Kit w/ Basler dart camera
#  Started from opencv.py @ github.com/basler/pypylon/samples
#  SJM / MCL 14-10-18
#
######################################################################

# --- check environment ---

import os
if os.name == 'nt':
  print('!WINDOWS: no syntax errors')
  raise SystemExit

# --- imports ---

from pypylon import pylon
import numpy as np
import cv2
import os, subprocess, time
import UKRegOCR
from pprint import pprint

# --- globals ---

TgtWH = (170,50)  # Nominal width and height in pixels for a number-plate in the camera's field of view

# --- support functions ---

# close and restart this script on demand, eg if an update has occurred
def restartMe():
  subprocess.call('sleep 2 && /usr/bin/python3 ' + __file__ + ' &', shell=True)
  print('Restarting in 2 secs...')
  raise SystemExit(0)

# --- MAIN ---

# get time-stamp of the script's source file
myTStamp = os.path.getmtime(__file__)

# choose a font for cv2 annotation
font = cv2.FONT_HERSHEY_SIMPLEX

# connect to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
print("Using camera:", camera.GetDeviceInfo().GetFriendlyName())

# Grab Continuous & convert to OpenCV BGR
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
  grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
  if grabResult.GrabSucceeded():
    img = converter.Convert(grabResult).GetArray() # convert to a cv2 image
    H,W = img.shape[0:2]  # get image dimensions
    captn = time.asctime()[11:19]
    x,y,w,h = UKRegOCR.lookForPlate(img, TgtWH)  # is there a number plate?
    if w * h > 0:
      # expand the bounding box & convert from XYWH => X1Y1X2Y2
      x1,y1 = max(0, x - w//2), max(0, y - h//2)
      x2,y2 = min(W-1, x + 3*w//2), min(H-1, y + 3*h//2)
      cv2.rectangle(img, (x1-1,y1-1), (x2,y2), (32,32,220), 5)  # draw red box
      # attempt an OCR read
      prr = UKRegOCR.ReadPlate(img[y1:y2, x1:x2], TgtWH)
      if (type(prr) is tuple) and (len(prr) == 4):
        reg,c,bbx,col = prr
        captn += " {:s} conf {:2.0f}% colour {:s} ({:d}x{:d})".format(reg, c, col, bbx[2], bbx[3])
        print(captn)
    # add caption
    cv2.putText(img,captn,(5,70), font, 3,(255,255,255),6,cv2.LINE_AA)
    # scale & display live image
    thmb = cv2.resize(img, (W//4, H//4))
    cv2.imshow('live', thmb)

    # quit on ESC/CR/update
    k = cv2.waitKey(1)
    if k == 27: break  # stop & re-start
    if k == 13: break  # stop
    # auto-restart on update
    k = 13
    if myTStamp != os.path.getmtime(__file__): break
  grabResult.Release()
    
# Release resources 
camera.StopGrabbing()
cv2.destroyAllWindows()

# re-start?
if k == 13: restartMe()
