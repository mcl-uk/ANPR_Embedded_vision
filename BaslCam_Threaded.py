######################################################################
#
#  ANPR Demo for UP Embedded Vision Starter Kit w/ Basler dart camera
#  Started from opencv.py @ github.com/basler/pypylon/samples
#  SJM / MCL 14-10-18
#
######################################################################

# 27-10-18 Add threading for OCR task 


# --- check environment ---

import os
if os.name == 'nt':
  print('!WINDOWS: no syntax errors')
  raise SystemExit

# --- imports ---

import os, subprocess, time, threading
from pprint import pprint
from pypylon import pylon
import numpy as np
import cv2
import UKRegOCR   # download from MarvellConsultants.co.uk

# --- globals ---

TgtWH = (170,50)  # Nominal width and height in pixels for a number-plate in the camera's field of view
max_threads = 2   # Maximum number of concurrent OCR threads
                  #  Note: would expect to select 3 here, for a 4 core processor
                  #  but 3 causes occasional RTEs, 2 seems to work much better

# --- support functions ---

# close and restart this script on demand, eg if an update has occurred
def restartMe():
  subprocess.call('sleep 1 && /usr/bin/python3 ' + __file__ + ' &', shell=True)
  print('Restarting...')
  raise SystemExit(0)

# threaded OCR wrapper
def doOCR(img, tgtwh, plateXYWH):
  ts = threading.current_thread().name
  prr = UKRegOCR.ReadPlate(img, tgtwh, plateXYWH)
  if (type(prr) is tuple) and (len(prr) == 4):
    reg,c,bbx,col = prr
    msg = "--- {:s} {:s} conf {:2.0f}% colour {:s} ({:d}x{:d})".format(ts, reg, c, col, bbx[2], bbx[3])
  else:
    msg = "--- {:s} unreadable".format(ts)
  print(msg)
  
# ascii unix time-stamp to Axis-style file-ref
def tref2ref(tref):
  t = time.localtime(float(tref))
  op = str(t.tm_year)[2:4] + '-'
  op += '{:02d}'.format(t.tm_mon) + '-'
  op += '{:02d}'.format(t.tm_mday) + '_'
  op += '{:02d}'.format(t.tm_hour) + '-'
  op += '{:02d}'.format(t.tm_min) + '-'
  op += '{:02d}'.format(t.tm_sec) + '-'
  op += '{:02d}'.format(int(float(tref) * 100.0) % 100)
  return op

# return a tuple listing all threads that aren't the Main thread
def listNonMainThreads():
  ths = ()
  for t in threading.enumerate():
    tn = t.getName()
    # dont wait for main thread
    if tn in ('MainThread', ): continue
    ths += (tn, )
  return ths

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
  time.sleep(0.005)
  grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
  if grabResult.GrabSucceeded():
    ts = str(time.time())
    img = converter.Convert(grabResult).GetArray() # convert to a cv2 image
    H,W = img.shape[0:2]  # get image dimensions
    iref = tref2ref(ts)
    captn = iref
    x,y,w,h = UKRegOCR.lookForPlate(img, TgtWH)  # is there a number plate?
    if w * h > 0:
      captn += " plate detected"
      # expand the bounding box & convert from XYWH => X1Y1X2Y2
      x1,y1 = max(0, x - w//2), max(0, y - h//2)
      x2,y2 = min(W-1, x + 3*w//2), min(H-1, y + 3*h//2)
      # Thread-out an OCR read process IF we have capacity
      if len(listNonMainThreads()) < max_threads:
        th = threading.Thread( target=doOCR, name=iref, args=(img, TgtWH, (x, y, w, h)) )
        th.start()
        captn += " submitted for OCR"
      # draw red box around suspected plate area
      cv2.rectangle(img, (x1-1,y1-1), (x2,y2), (32,32,220), 5)
    # add caption
    cv2.putText(img, iref, (5,70), font, 3, (255,255,255), 6, cv2.LINE_AA)
    # scale & display live image
    cv2.imshow('live', cv2.resize(img, (W//4, H//4)))
    print(captn)

    # quit on ESC/CR/update
    k = cv2.waitKey(1)
    if k == 27: break  # stop & re-start
    if k == 13: break  # stop
    # auto-restart on update
    k = 13
    if myTStamp != os.path.getmtime(__file__): break
  
  grabResult.Release()

print('Shutting down, wait for threads to terminate...')

# Release resources 
camera.StopGrabbing()

# Wait for threads to complete
cnt = 1
ttot = 0
while (cnt > 0) and (ttot < 200): # wait up to 20 seconds
  time.sleep(0.1)
  ttot += 1
  cnt = len(listNonMainThreads())
if (cnt > 0): print('Zombie threads:' + str(ths), 'wont terminate')

# shut windows
cv2.destroyAllWindows()

# exit or re-start?
if k == 13: restartMe()
