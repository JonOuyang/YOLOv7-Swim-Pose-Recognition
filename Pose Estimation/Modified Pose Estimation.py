#This is the pose estimation file to run (includes automatic rotation alg)
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
    global fc, c, t, nK, lK, na, nc, e, v
    output = non_max_suppression_kpt(output, 
                                     0.05, # Confidence Threshold
                                     0.25, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    #0.03, 0.2
    with torch.no_grad():
        output = output_to_keypoint(output)
        #print(f'Frame Number: {fc}; Data Size: {output.shape}')
    try:
        t = output[0] #retrieves only first skeleton data
        #t = t[-51:] #retrieves last 51 elements
        t = t[-36:]
        #append all nose x coords
        nK.append(t[0])
        #appends all hip x coords
        lK.append(t[18]) #array t is unsorted, the x values is every other starting at index 0 and y values every other starting at 1
        #index 22 gives the x coordinate for right hip joint
        #EXPERIMENTAL ADJUSTMENTS TO REMOVE NULL COORDINATES BASED ON A THRESHOLD VALUE
        for i in range(0, len(t), 3):
            g = t[i:i+3]
            if g[2] <= 0.20:
                t[i] = 0
                t[i+1] = 0
                v+=1
                e+=1
        t = [x for i, x in enumerate(t) if (i+1)%3 != 0]    #keypoint array without confidence levels
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
def swimPose_estimate(filename, savepath):
    global fc, c, t, nK, lK, na, fa, e, v
    
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
        v=0
        poseH(filename, "none", i*32)
        #print(f'Original data: {c} empty frames')
        cap.release()
        cv2.destroyAllWindows()

        if statistics.median(nK) > statistics.median(lK):    #compares median values of x coordinates of hips and shoulders
            na = []
            v=0
            poseH(filename, "cc", i*32)    #counterclockwise rotation
            if c <= 5:    #if there are less than 5 empty frames, append 
                z = np.array(na)
                z = np.reshape(z, (z.shape[0], 12, 2))
                fa.extend(rotate(z, (-90)))
            else:    #otherwise, discard
                print("too many missing frames, batch discarded")
        else:
            na = []
            poseH(filename, "c", i*32)    #clockwise rotation
            if c <= 5:
                z = np.array(na)
                z = np.reshape(z, (z.shape[0], 12, 2))
                fa.extend(rotate(z, 90))
            else:
                print("too many missing frames, batch discarded")
                
        i += 1    #batch counter (batches of __ frames)
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
    global fc, c, t, nK, lK, na
    cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Free_Skel.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
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
    cv2.destroyAllWindows()


def rotate(coordinates, ang):
    #coordinates=np.reshape(coordinates, (coordinates.shape[0], 24))
    ###if clockwise, (1280-y, original x coordinate) & if counterclockwise, (y, 720-original x)
    rotated_coordinates = []
    for i in coordinates:
        in_r = []
        for j in i:
            if ang==90:
                in_r.append([1280-j[1], j[0]])
            elif ang==-90:
                in_r.append([j[1], 720-j[0]])
        rotated_coordinates.append(in_r)
    return rotated_coordinates

#replace this with whatever your video path and save path is
video = "Testing Sample 2.mp4"
path = 'test.npy'
swimPose_estimate(video, path)
