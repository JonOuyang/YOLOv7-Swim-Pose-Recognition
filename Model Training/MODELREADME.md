### Base Model Trainig
The individual models are trained in this file. Currently the code is for ResNet moedls, but the other CNN and MLP models can be found in the Model Storage file. I used K-fold cross validation to train the models, and I used a somewhat questionable method to get my models to train longer. I repeatedly trained the same model on K-Folds multiple times (about 5 cycles, but you can change that parameter). I also used random seed for the model training starting point so you will see variation from 30% accuracy all the way up to 80% accuracy. I just kept the models that reached this 80% accuracy and discarded the others. 

### Hierarchical Models
I assembled the models into the tree as outlined in my paper. I left this file as a Jupyter Notebook file because there's some notes at the bottom, where I recorded the highest ensemble accuracies. Unfortunately, it does not look like I recorded the name of the models used, but I should have all of them still.

### Preprocessing
The preprocessing just batched the data into groups of 32 (frames) and applied augmentations like shuffling, mirror, and noise injections. If you're interested, there's code for batching into groups of 64 and 128 frames in a different Github repo from a paper I worked on before this one (https://github.com/JonOuyang/CNN-Exercise-Recognition/blob/main/preprocessing.py). The main hbatching method is the same across all groups

### Model Storage
I didn't want to make my main files too messsy with model architectures commented out so I put them all in a separate file. There should be a couple comments lying around describing some observations for each model. Again, refer to main paper and main file for which models to use. Not all of the models in the storage were effective.
