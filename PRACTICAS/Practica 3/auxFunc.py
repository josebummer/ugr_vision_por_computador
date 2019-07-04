# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:27:33 2017

@author: Nicolás Pérez de la Blanca
   """
import cv2
import pickle
import numpy as np
#from matplotlib import pyplot as plt

def click_and_draw(event,x,y,flags,param):
    global refPt, imagen,FlagEND
    
    
   # if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if  event == cv2.EVENT_LBUTTONDBLCLK:
        FlagEND= False
        cv2.destroyWindow("image")
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        #cropping = True
        print("rfePt[0]",refPt[0])
    

    elif (event == cv2.EVENT_MOUSEMOVE) & (len(refPt) > 0) & FlagEND:
    # check to see if the mouse move
        clone=imagen.copy()
        nPt=(x,y)
        print("npt",nPt)
        sz=len(refPt)
        cv2.line(clone,refPt[sz-1],nPt,(0, 255, 0), 2)
        cv2.imshow("image", clone)
        cv2.waitKey(0)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        #cropping = False
        sz=len(refPt)
        print("refPt[sz]",sz,refPt[sz-1])
        cv2.line(imagen,refPt[sz-2],refPt[sz-1],(0, 255, 0), 2)
        cv2.imshow("image", imagen)
        cv2.waitKey(0)
        

def extractRegion(image):
    global refPt, imagen,FlagEND
    imagen=image.copy()
    # load the image and setup the mouse callback function
    refPt=[]
    FlagEND=True
    #image = cv2.imread(filename)
    cv2.namedWindow("image")
    # keep looping until the 'q' key is pressed
    cv2.setMouseCallback("image", click_and_draw)
    #
    while FlagEND:
    	# display the image and wait for a keypress
        cv2.imshow("image", image)
        cv2.waitKey(0)
    #
    print('FlagEND', FlagEND)
    refPt.pop()
    refPt.append(refPt[0])
    cv2.destroyWindow("image")
    return refPt	


def loadDictionary(filename):
    with open(filename,"rb") as fd:
        feat=pickle.load(fd)
    return feat["accuracy"],feat["labels"], feat["dictionary"]

def loadAux(filename, flagPatches):
    if flagPatches:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"],feat["patches"]
    else:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"]


