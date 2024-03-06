#YOLO uses PyTorch for model
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
fc = 1    #frame count
na = []    #final coordinate array (with confidence levels)
kk=[]    #final coordinate array (with confidence levels removed
c=0    #number of empty frames (no keypoints detected)
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
model = load_model()    #load model weights

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
    global fc, c, na, kk
    output = non_max_suppression_kpt(output, 
                                     0.03, # Confidence Threshold
                                     0.2, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
        #print(f'Frame Number: {fc}; Data Size: {output.shape}')
    try:
        t = output[0] #retrieves only first skeleton data
        t = t[-51:] #retrieves last 51 elements
        #t = t[::3] cuts every third element (confidence level)
        na.append(t)
        #append confidence levels respectively
        z= [x for i, x in enumerate(t) if (i+1)%3 != 0]    #cuts every third element (confidence level)
        kk.append(z)
        
    except:
        c += 1    #add 1 to empty frame number
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return nimg

def pose_estimation_video(filename):
    global fc, g
    cap = cv2.VideoCapture(filename)    #it's standard convention to call the video file "cap" when using cv2
    totalFrames = math.floor(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))    #get the number of frames in video
    print(f'Total Frame count: {totalFrames}')    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')    #video output type 
    out = cv2.VideoWriter('t2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(frame)    #run model using this frame as input, predict joint coordinate keypoints
            frame = draw_keypoints(output, frame)    #take the predicted coordinates and map them onto the image
            fc += 1     #add 1 to frame count
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))    
            frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)    #resize frame (the interpolation was part of the reference code)
            out.write(frame)
            cv2.imshow('Pose estimation', frame)    #display frame on a new window (.imshow will run for every single frame. your framerate should be about 10-20fps i think?)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):    #This is supposed to stop the entire program if you press 'q', but it doesn't work
            break
    cap.release()    #algorithm finish, disband variable
    out.release()    #algorithm finish, disband variable
    cv2.destroyAllWindows()    #close window the video was playing on

videoPath = "file path"    #put whatever file path you want to save the video to
#pose_estimation_video(video)

def preprocess(videoPath, savePath):
    global fc, c, na, skelData
    c=0
    video = videoPath
    na = []    #clears array
    #I didn't do anything with the array containing coordinate confidence. If you want to do something with that array, the array name is "kk"
    pose_estimation_video(video)
    skelData = np.array(na)    #convert to numpy array for faster loading (your computer might literally crash if you don't convert to numpy)
    skelData = skelData.reshape(skelData.shape[0], 17, 2)    #reshape array so that they look like (x, y) coordinates
    print(skelData.shape)
    print(f'{c} total empty frames')
    print(skelData[0])
    np.save(savePath, skelData)    #save to path

print(f'{c} frames of empty data')
