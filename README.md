# YOLOv7-Swim-Pose-Recognition

Our research has been presented at the 2024 IEEE Southwest Symposium on Image Analysis and Interpretation, held in Sante Fe, New Mexico on March 18th!

#### Abstract:
Swim pose estimation and recognition is a challenging problem in Machine Learning (ML) and Artificial Intelligence (AI) as the body of the swimmer is continuously submerged under the water. The objective of this paper is to enhance existing ML models for estimating swim poses and for recognizing strokes to aide swimmers in pursuing a more perfect technique. We developed a novel methodology augmenting raw video data and adjusting a YOLOv7 base model to enhance swim pose estimation. We found the standard multi-class classification Convolutional Neural Network (CNN) to be insufficient for stroke recognition due to the similarity between strokes, so we designed a hierarchical binary classification tree using multiple ensembles of multilayer perceptron (MLP), CNN, and residual network (ResNET) models. Through these optimizations, the confidence level of pose estimation has increased by over 30%, and the ensembles of our recognition model has achieved approximately 80% accuracy. Fine-tuning of our recognition models and research combining joint keypoint coordinates with angle measurements as inputs could further increase the accuracy of our models.

NOTE: In swimming terms, "long axis strokes" refer to Freestyle and Backstroke; "short axis strokes" refer to Butterfly and Breastroke

# Github Repository Notes:
**THIS REPOSITORY IS NOT MEANT TO BE CLONED. YOU MUST INSTALL THE LIBRARIES YOURSELF, INCLUDING YOLO POSE ESTIMATION**
### Libraries Required:
(NOTE: since this project was highly experimental, not all of the libraries imported in the file were used)
- YOLOv7 (NOTE: during the duration of the research YOLOv7 was the most recent version available. However, YOLOv8 was released in the month following its completion, so if you would like to upgrade the YOLO version you may have to research the implementation separately. Besides the Yolov7 Pose file everything should work the same since they are not reliant on YOLO. Also, importing YOLO should import a lot of the 
- Numpy
- Tensorflow (model training)
- sklearn
- matplotlib
- cv2

## Keypoint
This folder contains details on the exact indices of each joint keypoint that is returned from the pose estimation algorithm. For some reason, when you google this information there's a lot of contraditing info and diagrams so for your convenience, I've verified these keypoints myself. There is also a [visual diagram](Visual.png) for them.

## Models
This folder contains all of the files for building and training the neural networks. See the README file in the Model Training folder for more info.
If one of the ipynb files (Jupyter Notebooks) get cut off or don't fully load, reload the page.

## Pose Estimation
1. YOLOv7 Pose.py - This is the original pose estimation model that I referenced. Requires YOLOv7 to be installed on your local machine. There might be some small bugs in that file since I modified it a bit in the github repo.
2. Modified Pose Estimation.py - This is the pose estimation model that I modified from the original. It includes the automatic rotation algorithm and some nice print statements to track progress.
3. Paper Data.py - This is almost the exact same thing as Modified Pose Estimation.py, but it includes more data collection. It collects data from the original (unrotated) video pose estimation and data from the new rotated video pose estimation. It generates graphs at the end, which is what I used to make the charts that are included in my paper. Useful for writing my paper, irrelevant for the actual development. 

## Other
- The Main.ipynb file is the entire project put together, built off of files and networks from other files.
- Feedback Functions.py contains some basic functions that compare the angles and gives feedback on swim technique.
- Spare Parts.py are random parts that I didn't want to keep in my main files because it would just make things messy. So they're stored in here.

# Overview
## 1. The Recognition Model
  - Three primary model architectures were tested on the initial dataset:
        - 2D CNNs
        - 3D CNNs (2D casted into 3D where third dimension represents left vs right)
        - Multilayer Perceptron Networks (MLP)
        - ResNet with 2D Convolution layers
        - ResNet with 3D Convlution layers
    - 3 unique datasets have been created, all with roughly equal divisions in class.
    - The dataset includes 6 classes: Freestyle, Butterfly, Breaststroke, Backstroke, Underwater (Dolphin/Breastroke), and Diving
    - Frame batches of 32, 64, 128 were tested
    - 12 joint coordinate keypoints are used, all head keypoints are excluded through feature extraction. Encountered issues with head keypoints interfering with model accuracies, so head keypoints were removed through feature extraction.
    *More details can be found in the actual code itself.

### Classification Tree
1. The Recognition Tree consists of 2 primary layers. At the top layer, the input in form (1, 32, 12, 3) is fed into a binary classification hard voting ensemble, where 5 models guess whether the class is long axis vs short axis stroke, each guess counting as a single unweighted "vote". The majority of votes represents the final class that is guessed.
2. The second layer after the initial long axis vs short axis stroke recognition is 2 separate paths.
     a. The first path is the long axis path, where there is a singular DNN model to differentiate Freestyle vs Backstroke. Voting Ensemble in preliminary testing seemed to have no affect on accuracy. The 
     b. The second path is the short axis path, where there is a Soft Voting Ensemble, where 5 models returns its predictions in the form of confidence level (i.e. [0.7, 0.3] class 0 prediction), and the average confidence level for each class is calculated and the class with the overall highest confidence level is counted as the final guess. Significant improvement in accuracy was recorded.
3. After classification of stroke, the program will then execute its respective feedback functions, taking in the input as parameter for analysis
   
## 2. The Feedback Algorithm
- The Feedback Algorithm will not use a ML-based approach, instead, will utilize the keypoints drawn by YOLOv7 to compare angles and relative positioning between joints.
  - Correct joint positioning will be hard coded, with an acceptable and unacceptable range.
- The feedback functions are based on a calculated angle (degrees) between three joint coordinates
  * this project does not focus on this portion
