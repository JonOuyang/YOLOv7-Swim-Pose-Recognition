import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import statistics

import math

"""
I'm so sorry about the ambiguous variable names. I didn't make proper plans to publish my code for others to read.
fc - frame count
na - new array?
actualKp - actual key point (array)
c - idk
lK - something keypoint
nK - nose keypoint
la - something array
fullArray - takes the batches of frames and combines them into a single array
initialArray - for storing initial values with unaugmented data
batchArray - stores new values with augmented data in batch and appends to full array
tc - total count?
"""

fc = 1
na = []
actualKp = []
c=0
lK = []
nK = []
la = []
fullArray = []
initialArray = []
batchArray = []
tc = 0

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
    global fc, c, t, nK, lK, na, nc, e, v, batchArray, fullArray, initialArray
    output = non_max_suppression_kpt(output, 
                                     0.03, # Confidence Threshold
                                     0.2, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    #0.03, 0.2
    with torch.no_grad():
        output = output_to_keypoint(output)
        #print(f'Frame Number: {fc}; Data Size: {output.shape}')
    try:
        t = output[0] #retrieves only first skeleton data
        batchArray.append(t[-51:])#retrieves last 51 elements
        t = t[-36:]
        #append all nose x coords
        nK.append(t[0])
        #appends all hip x coords
        lK.append(t[18]) #array t is unsorted, the x values is every other starting at index 0 and y values every other starting at 1
        #index 22 gives the x coordinate for right hip joint
        
        t = [x for i, x in enumerate(t) if (i+1)%3 != 0]
        #cuts confidence level from array
        na.append(t)
    except:
        c += 1
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    
    
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    
    return nimg

nc = 0
fa = []
e=0
v=0
initialArray = []

def swimPose_estimate(filename, savepath):
    global f, fc, c, t, nK, lK, na, fa, e, v, conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16, tc
    
    cap = cv2.VideoCapture(filename)
    totalFrames = math.floor(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/32)
    print(f'TF: {totalFrames}')
    i = 0
    fa = []
    e=0
    
    while i < totalFrames:
        na = []
        fc = 0
        nK = []
        lK = []
        c=0
        poseH(filename, "none", i*32)
        print(f'Original data: {c} empty frames')
        #append to original frame confidence chart
        for u in batchArray:
            initialArray.append(u)

        cap.release()
        cv2.destroyAllWindows()
       
        #Perm Rotations
        fc = 0
        c=0
        v=0
        if statistics.median(nK) > statistics.median(lK):
            na = []
            v=0
            #print("needs counterclockwise rotation")
            poseH(filename, "cc", i*32)
            #print("transformation completed")
            print(f'Counterclockwise rotation; {c} empty frames')
            z = np.array(na)
            z = np.reshape(z, (z.shape[0], 12, 2))
            fa.extend(rotate(z, (-90)))
            for u in batchArray:
                fullArray.append(u)

        else:
            na = []
            #print("needs clockwise rotation")
            poseH(filename, "c", i*32)
            #print("transformation completed")
            print(f'Clockwise rotation; {c} empty frames')
            z = np.array(na)
            z = np.reshape(z, (z.shape[0], 12, 2))
            fa.extend(rotate(z, 90))
            for u in batchArray:
                fullArray.append(u)
            #print(v)
            
        
        tc += c    
        i += 1
        print(f'batch {i} complete')
        
    print("=======================================================")
    print("-----------Skeleton Data Extraction Complete-----------")
    print("=======================================================")

    x = np.array(fa)
    x = np.reshape(x, (x.shape[0], 12, 2))
    print(f'Array shape: {x.shape}')
    np.save(savepath, x)
    print(f'Data saved to: {savepath}')
    print(f'{e} total coordinates voided')

    print("=======================================================")
    cv2.destroyAllWindows()


def poseH(filename, rotation, currentFrame):
    global fc, c, t, nK, lK, na, batchArray, fullArray
    #cv2.destroyAllWindows()

    cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Free_Skel.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
    batchArray = []
    while fc < 32 and cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotation == "cc":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == "c":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == "none":
                pass

            output, frame = run_inference(frame)
            frame = draw_keypoints(output, frame)
            fc += 1
            
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            
            if rotation == "cc" or rotation == "c":
                frame = cv2.resize(frame,(720,1280),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            else:
                frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    #cv2.destroyAllWindows()


def rotate(coordinates, ang):
    #coordinates = np.array(coords)
    #angles = np.random.uniform(low=-max_angle, high=max_angle)
    angles = np.deg2rad(ang)
    center = np.mean(coordinates, axis=(0, 1))  # Compute the center of rotation

    rotation_matrix = np.array([[np.cos(angles), -np.sin(angles)],
                                [np.sin(angles), np.cos(angles)]])

    rotated_coordinates = np.zeros_like(coordinates)

    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            # Translate coordinates to the center of rotation
            translated_coord = coordinates[i, j] - center

            # Apply rotation to the translated coordinates
            rotated_coord = np.dot(rotation_matrix, translated_coord.T).T

            # Translate back to the original position
            rotated_coordinates[i, j] = rotated_coord + center

    rotated_coordinates = rotated_coordinates.tolist()
    return rotated_coordinates

%%time
video = "C:/Users/jonso/OneDrive/Desktop/Breast Training Data 2.mp4"
path = 'nt.npy'
swimPose_estimate(video, path)

#the following conf0-16 is for the new augmented data values to plot
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

#fullArray = fullArray[1::2]
fullArray = np.array(fullArray)
for t in fullArray:
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


print(f'{tc} frames of empty data')

conf0i = []
conf1i = []
conf2i = []
conf3i = []
conf4i = []
conf5i = []
conf6i = []
conf7i = []
conf8i = []
conf9i = []
conf10i = []
conf11i = []
conf12i = []
conf13i = []
conf14i = []
conf15i = []
conf16i = []

#conf0i-16i is for initial values plotting
initialArray = np.array(initialArray)
for t in initialArray:
    conf0i.append(t[2])
    conf1i.append(t[5])
    conf2i.append(t[8])
    conf3i.append(t[11])
    conf4i.append(t[14])
    conf5i.append(t[17])
    conf6i.append(t[20])
    conf7i.append(t[23])
    conf8i.append(t[26])
    conf9i.append(t[29])
    conf10i.append(t[32])
    conf11i.append(t[35])
    conf12i.append(t[38])
    conf13i.append(t[41])
    conf14i.append(t[44])
    conf15i.append(t[47])
    conf16i.append(t[50])
        
conf0i = np.array(conf0i)
conf1i = np.array(conf1i)
conf2i = np.array(conf2i)
conf3i = np.array(conf3i)
conf4i = np.array(conf4i)
conf5i = np.array(conf5i)
conf6i = np.array(conf6i)
conf7i = np.array(conf7i)
conf8i = np.array(conf8i)
conf9i = np.array(conf9i)
conf10i = np.array(conf10i)
conf11i = np.array(conf11i)
conf12i = np.array(conf12i)
conf13i = np.array(conf13i)
conf14i = np.array(conf14i)
conf15i = np.array(conf15i)
conf16i = np.array(conf16i) 


print(f'{tc} frames of empty data')



import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

#plt.figure()
totalConfFinal = np.concatenate((conf5, conf6, conf7, conf8, conf9, conf10, conf11, conf12, conf13, conf14, conf15, conf16))
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(totalConfFinal, bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
ax.set_ylim([0, 10000])
ax.set_xlabel("Confidence Levels of Predictions")
ax.set_ylabel("Number of measurements in region")
plt.show()

#plt.figure()
totalConfInitial = np.concatenate((conf5i, conf6i, conf7i, conf8i, conf9i, conf10i, conf11i, conf12i, conf13i, conf14i, conf15i, conf16i))
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(totalConfInitial, bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
#ax.set_ylim([0, 10000])
ax.set_ylim([0, 10000])
ax.set_xlabel("Confidence Levels of Predictions")
ax.set_ylabel("Number of measurements in region")
plt.show()

print(np.median(totalConfFinal))
