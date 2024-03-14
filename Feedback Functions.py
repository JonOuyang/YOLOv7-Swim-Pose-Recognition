import numpy as np
import tensorflow as tf

import datetime
import os
import matplotlib.pyplot as plt

import math

num_classes=4

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x2_test.npy')
y_test = np.load('y2_test.npy')

#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
 
x = x_train[0][0]

class dive:
    def hips(data):
        print(data[6][1])
        return 0 if min(data[5][1], data[6][1]) <= min(data[0][1], data[1][1]) else 1
    def kneeAngle(data):
        return 0 if (90 < angle3pt(data[10], data[8], data[6]) < 270) or (90 < angle3pt(data[11], data[9], data[7]) < 270) else 1

class underwater:
    def legsTogether(data):
        return 0 if not ((30 < angle3pt(data[8], data[6], data[9]) < 330) or (30 < angle3pt(data[8], data[7], data[9]) < 330)) else 1

class freestyle:
    def elbowDrop(data):
        return 0 if not ((60 < angle3pt(data[10], data[8], data[6]) < 300) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1
    def kneeAngle(data):
        #looks at the angle of the knee, from hip to knee to ankle
        return 0 if ((70 < angle3pt(data[6], data[8], data[10]) < 290) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1
    def sinkHip(data):
        #looks at angle of shoulder to hip to knee to check if hips are sinking too low in water
        return 0 if not ((130 < angle3pt(data[0], data[6], data[8] < 230)) or (130 < angle3pt(data[1], data[7], data[9]) < 230)) else 1

class backstroke:
    def kneeAngle(data):
        return 0 if ((70 < angle3pt(data[6], data[8], data[10]) < 290) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1
    def sinkHip(data):
        return 0 if not ((130 < angle3pt(data[0], data[6], data[8] < 230)) or (130 < angle3pt(data[1], data[7], data[9]) < 230)) else 1
    def straightArm(data):
        return 1 if ((170 < angle3pt(data[0], data[2], data[4]) < 190) or (170 < angle3pt(data[1], data[3], data[5]) < 190)) else  0 

class butterfly:
    def elbowDrop(data):
        return 0 if not ((60 < angle3pt(data[10], data[8], data[6]) < 300) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1
    def kickAngle(data):
        return 0 if ((70 < angle3pt(data[6], data[8], data[10]) < 290) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1
    def chestDown(data):
        return 0 if min(data[0][1], data[1][1]) < min(data[6][1], data[7][1]) else 1
    def legsTogether(data):
        return 0 if not ((30 < angle3pt(data[8], data[6], data[9]) < 330) or (30 < angle3pt(data[8], data[7], data[9]) < 330)) else 1
      
class breastroke:
    def noKick(data):
        return 0 if ((70 < angle3pt(data[6], data[8], data[10]) < 290) or (90 < angle3pt(data[11], data[9], data[7]) < 270)) else 1

#e1-e(i) represents error # __
if class==0:
    e1 = 0
    e2 = 0
    e3 = 0
    for frame in x:
        e1 += 1 if freestyle.elbowDrop(frame) == 0 else 0
        e2 += 1 if freestyle.kneeAngle(frame) == 0 else 0
        e3 += 1 if freestyle.sinkHip(frame) == 0 else 0
elif class == 1:
    e1 = 0
    e2 = 0
    e3 = 0
    for frame in x:
        e1 += 1 if backstroke.kneeAngle(frame) == 0 else 0
        e2 += 1 if  backstroke.sinkHip(frame) == 0 else 0
        e3 += 1 if backstroke.straightArm(frame) == 1 else 0
elif class == 2:
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0
    for frame in x:
        e1 += 1 if butterfly.elbowDrop(frame) == 0 else 0
        e2 += 1 if butterfly.kickAngle(frame) == 0 else 0
        e3 += 1 if butterfly.chestDown(frame) == 0 else 0
        e4 += 1 if butterfly.legsTogether(frame) == 0 else 0
elif class == 3:
    e1 = 0
    for frame in x:
        e1 += 1 if breastroke.noKick(frame) == 0 else 0
