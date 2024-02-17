import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
fc = 1
#na = np.empty((1, 51))
na = []
kk=[]
c=0
#confidence trackers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()

def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image

def draw_keypoints(output, image):
    global fc, c, na, kk, conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16
    output = non_max_suppression_kpt(output, 
                                     0.03, # Confidence Threshold
                                     0.2, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    #0.02, 0.9
    #0.03, 0.2
    with torch.no_grad():
        output = output_to_keypoint(output)
        #print(f'Frame Number: {fc}; Data Size: {output.shape}')
    singleSkel = []
    try:
        t = output[0] #retrieves only first skeleton data
        t = t[-51:] #retrieves last 51 elements
        #t = t[::3] cuts every third element (confidence level)
        na.append(t)
        #append confidence levels respectively
        
        #fin
        z= [x for i, x in enumerate(t) if (i+1)%3 != 0]
        kk.append(z)
        
    except:
        #na.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,])
        c += 1
        #na.append([null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null,null, null])
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    
    
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    
    
        
    return nimg

def pose_estimation_video(filename):
    global fc, g, conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16
    
    conf0 = []
    conf1 = []
    conf2 = []
    conf3 = []
    conf4 = []
    conf5 = []
    conf6 = []
    conf7 = []
    conf8 = []
    conf9 = []
    conf10 = []
    conf11 = []
    conf12 = []
    conf13 = []
    conf14 = []
    conf15 = []
    conf16 = []

    cap = cv2.VideoCapture(filename)
    totalFrames = math.floor(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f'Total Frame count: {totalFrames}')    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('t2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            output, frame = run_inference(frame)
            frame = draw_keypoints(output, frame)
            fc += 1 
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            """p = np.array(na[-1])
            p = np.reshape(p, (17, 3))
            g = np.array(kk[-1])
            g = np.reshape(g, (17, 2))
            print(p)
            print("-=-=-=-=-=-=-=-=-=")
            print(g)
            print(c)"""
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()



video = "C:/Users/jonso/OneDrive/Desktop/Dive Training Data 2.mp4"
pose_estimation_video(video)

na = np.array(na)
for t in na:
    conf0.append(t[2])
    conf1.append(t[5])
    conf2.append(t[8])
    conf3.append(t[11])
    conf4.append(t[14])
    conf5.append(t[17])
    conf6.append(t[20])
    conf7.append(t[23])
    conf8.append(t[26])
    conf9.append(t[29])
    conf10.append(t[32])
    conf11.append(t[35])
    conf12.append(t[38])
    conf13.append(t[41])
    conf14.append(t[44])
    conf15.append(t[47])
    conf16.append(t[50])
        
conf0 = np.array(conf0)
conf1 = np.array(conf1)
conf2 = np.array(conf2)
conf3 = np.array(conf3)
conf4 = np.array(conf4)
conf5 = np.array(conf5)
conf6 = np.array(conf6)
conf7 = np.array(conf7)
conf8 = np.array(conf8)
conf9 = np.array(conf9)
conf10 = np.array(conf10)
conf11 = np.array(conf11)
conf12 = np.array(conf12)
conf13 = np.array(conf13)
conf14 = np.array(conf14)
conf15 = np.array(conf15)
conf16 = np.array(conf16)        
        
print(na.shape)

def preprocess(videoPath, savePath):
    global fc, c, na, skelData
    c=0
    video = videoPath
    na = []
    pose_estimation_video(video)
    skelData = np.array(na)
    skelData = skelData.reshape(skelData.shape[0], 17, 2)
    print(skelData.shape)
    print(f'{c} total empty frames')
    #print(na)
    print(skelData[0])
    np.save(savePath, skelData)

print(f'{c} frames of empty data')

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
"""cArrays = [conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16]
c = 0
for i in cArrays:
    fig, ax = plt.subplots(figsize = (10, 7))
    ax.hist(i, bins = bins)
    ax.set_xlabel("Confidence Levels in %")
    ax.set_ylabel("Number of measurements in region")
    plt.title(f'graph {c}')
    c+=1
    plt.show()"""



totalConf = np.concatenate((conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16))
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(totalConf, bins = bins)
ax.set_xlabel("Confidence Levels of Predictions")
ax.set_ylabel("Number of Predictions")
plt.title(f'All Combined')
plt.show()



totalConf = np.concatenate((conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16))
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(totalConf, bins = bins)
#ax.set_ylim([0, 10000])
ax.set_ylim([0, 15000])
ax.set_xlabel("Confidence Levels of Predictions")
ax.set_ylabel("Number of Predictions")
plt.title(f'Excluding Facial Keypoints')
plt.show()
