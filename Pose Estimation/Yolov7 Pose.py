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
    singleSkel = []
    try:
        t = output[0] #retrieves only first skeleton data
        t = t[-51:] #retrieves last 51 elements
        #t = t[::3] cuts every third element (confidence level)
        na.append(t)
        #append confidence levels respectively
        z= [x for i, x in enumerate(t) if (i+1)%3 != 0]
        kk.append(z)
        
    except:
        c += 1
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return nimg

def pose_estimation_video(filename):
    global fc, g
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
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

video = "file path"
pose_estimation_video(video)

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
    print(skelData[0])
    np.save(savePath, skelData)

print(f'{c} frames of empty data')
