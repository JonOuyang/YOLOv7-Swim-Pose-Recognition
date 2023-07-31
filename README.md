# YOLOv7-Swim-Pose-Recognition
This project aims to utilize (2023) YOLOv7's new pretrained pose recognition algorithm for both recognition and feedback of swimming "poses" (strokes). 

There are two main sections of code:
1. The Recognition Model
  - As of Summer 2023, several models have been tested:
        - Basic 5-9 layer 2D CNNs (with 3-5 Convolution layers respectively)
        - Basic 5-9 layer 3D CNNs (with 3-5 Convolution layers respectively)
        - ResNet with 2D Convolution layers
        - ResNet with 3D Convlution layers
    - 3 unique datasets have been created, all with roughly equal divisions in class.
    - The dataset includes 6 classes: Freestyle, Butterfly, Breaststroke, Backstroke, Underwater (Dolphin/Breastroke), and Diving
    - Frame batches of 32, 64, 128 were tested
    *More details can be found in the actual code itself. Refer to Inference Model.ipynb

2. The Feedback Algorithm
- The Feedback Algorithm will not use a ML-based approach, instead, will utilize the keypoints drawn by YOLOv7 to compare angles and relative positioning between joints.
  - Correct joint positioning will be hard coded, with an acceptable and unacceptable range.

Contact me at:
jonsouyang@gmail.com
