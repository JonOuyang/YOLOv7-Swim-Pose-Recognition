# YOLOv7-Swim-Pose-Recognition
This project aims to utilize (2023) YOLOv7's new pretrained pose recognition algorithm for both recognition and feedback of swimming "poses" (strokes). 

##### NOTE: In swimming terms, "long axis strokes" refer to Freestyle and Backstroke; "short axis strokes" refer to Butterfly and Breastroke.
There are two main sections of code:
## 1. The Recognition Model
  - Three primary model architectures were tested on the initial dataset:
        - Deep 2D CNNs
        - Deep 3D CNNs
        - Dense Neural Networks (DNN)
        - ResNet with 2D Convolution layers
        - ResNet with 3D Convlution layers
    - 3 unique datasets have been created, all with roughly equal divisions in class.
    - The dataset includes 6 classes: Freestyle, Butterfly, Breaststroke, Backstroke, Underwater (Dolphin/Breastroke), and Diving
    - Frame batches of 32, 64, 128 were tested
    - 12 joint coordinate keypoints are used, all head keypoints are excluded for sake of accuracy. Encountered issues with head keypoints interfering with model accuracies.
    *More details can be found in the actual code itself. Refer to Inference Model.ipynb

### Classification Tree
1. The Recognition Tree consists of 2 primary layers. At the top layer, the input in form (1, 32, 12, 3) is fed into a binary classification hard voting ensemble, where 5 models guess whether the class is long axis vs short axis stroke, each guess counting as a single unweighted "vote". The majority of votes represents the final class that is guessed.
2. The second layer after the initial long axis vs short axis stroke recognition is 2 separate paths.
     a. The first path is the long axis path, where there is a singular DNN model to differentiate Freestyle vs Backstroke. Voting Ensemble in preliminary testing seemed to have no affect on accuracy. The 
     b. The second path is the short axis path, where there is a Soft Voting Ensemble, where 5 models returns its predictions in the form of confidence level (i.e. [0.7, 0.3] class 0 prediction), and the average confidence level for each class is calculated and the class with the overall highest confidence level is counted as the final guess. Significant improvement in accuracy was recorded.
3. After classification of stroke, the program will then execute its respective feedback functions, taking in the input as parameter for analysis
   
## 2. The Feedback Algorithm
- The Feedback Algorithm will not use a ML-based approach, instead, will utilize the keypoints drawn by YOLOv7 to compare angles and relative positioning between joints.
  - Correct joint positioning will be hard coded, with an acceptable and unacceptable range.
The feedback functions are based on a calculated angle (degrees) between three joint coordinates 

#Submissions
My work has been submitted to the following:
- REGENERON Science Talent Search (STS) 2024 [High School Research Competition]
- 2024 IEEE 6th International Conference on Artificial Intelligence Circuits and Systems (AICAS) [Conference Paper]
- 2024 IEEE Southwest Symposium on Image Analysis and Interpretation (SSIAI) [Conference Paper]
