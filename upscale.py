import cv2
import matplotlib.pyplot as plt
import time


def upscale_nn(im):
    img = im # cv2.imread(im)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
    path = "models/LapSRN_x4.pb"
 
    sr.readModel(path)
 
    sr.setModel("lapsrn",4)
 
    result = sr.upsample(img)
    return result
