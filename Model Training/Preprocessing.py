import numpy as np
import math
import statistics as st
from statistics import mode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


skel0 = np.load()
skel1 = np.load()
skel2 = np.load()
skel3 = np.load()
skel4 = np.load()
skel5 = np.load()

skel4 = np.concatenate((skel4, skel4, skel4))
skel5 = np.concatenate((skel5, skel5, skel5))

label0 = np.full((skel0.shape[0], 1), 0)
label1 = np.full((skel1.shape[0], 1), 0)
label2 = np.full((skel2.shape[0], 1), 1)
label3 = np.full((skel3.shape[0], 1), 1)
label4 = np.full((skel4.shape[0], 1), 4)
label5 = np.full((skel5.shape[0], 1), 5)

label4 = np.concatenate((label4, label4, label4))
label5 = np.concatenate((label5, label5, label5))
#print(label)


#y=0
nSkel = 0
nLabel = 0
def convert32(testSkel, testLabel, spSkel, spLabel):
    global y, nSkel, nLabel
    c=0
    print("=======================================================")
    print(f'original skel shape: {testSkel.shape}')
    print(f'original label shape: {testLabel.shape}')
    #testSkel = testSkel[:, 5:, :]
    nSkel = 0
    nLabel = 0
    while testSkel.shape[0]%32 != 0:
        testSkel = np.delete(testSkel, -1, axis=0)
        testLabel = np.delete(testLabel, -1, axis = 0)
        c += 1
    y = int(testSkel.shape[0]/16)-1
    nSkel = np.empty([y, 32, 12, 2])
    nLabel = np.empty([y, 1])
    for i in range(0, y):
        nSkel[i] = testSkel[(i*16):(i*16)+32,:,:]
        nLabel[i] = int(st.median(testLabel.flatten()[(i*16):(i*16)+32]))
    #experimental
    nSkel = nSkel[:200, :, :, :]
    nLabel = nLabel[:200, :]
    print(f'new shape (skel): {nSkel.shape}')
    print(f'new shape (label): {nLabel.shape}')
    print(f'A total of {c} elements have been removed')
    np.save(spLabel, nLabel)
    np.save(spSkel, nSkel)
    print("=======================================================")
    #return y, nSkel, nLabel
    
def convertO32(testSkel, testLabel, spSkel, spLabel):
    global y, nSkel, nLabel
    c=0
    print("=======================================================")
    print(f'original skel shape: {testSkel.shape}')
    print(f'original label shape: {testLabel.shape}')
    nSkel = 0
    nLabel = 0
    while testSkel.shape[0]%32 != 0:
        testSkel = np.delete(testSkel, -1, axis=0)
        testLabel = np.delete(testLabel, -1, axis = 0)
        c += 1
    y = int(testSkel.shape[0]/16)-1
    nSkel = np.empty([y, 32, 12, 2])
    nLabel = np.empty([y, 1])
    for i in range(0, y):
        nSkel[i] = testSkel[(i*16):(i*16)+32,:,:]
        nLabel[i] = int(st.median(testLabel.flatten()[(i*16):(i*16)+32]))
    print(f'new shape (skel): {nSkel.shape}')
    print(f'new shape (label): {nLabel.shape}')
    print(f'A total of {c} elements have been removed')
    np.save(spLabel, nLabel)
    np.save(spSkel, nSkel)
    print("=======================================================")
    #return y, nSkel, nLabel

convert32(skel0, label0, 'FreeSkel(c32).npy', 'FreeLabel(c32).npy')
convert32(skel1, label1, 'BackSkel(c32).npy', 'BackLabel(c32).npy')
convert32(skel2, label2, 'FlySkel(c32).npy', 'FlyLabel(c32).npy')
convert32(skel3, label3, 'BreastSkel(c32).npy', 'BreastLabel(c32).npy')
convert32(skel4, label4, 'UnderwaterSkel(c32).npy', 'UnderwaterLabel(c32).npy')
convert32(skel5, label5, 'DiveSkel(c32).npy', 'DiveLabel(c32).npy') 
convertO32(np.load('skel_test.npy'), np.load('label_test.npy'), 'x2_test.npy', 'y2_test.npy')


FreeSkel = np.load()
FlySkel = np.load()
BackSkel = np.load()
BreastSkel = np.load()
UnderwaterSkel = np.load()
DiveSkel = np.load()

FreeLabel = np.load()
FlyLabel = np.load()
BackLabel = np.load()
BreastLabel = np.load()
UnderwaterLabel = np.load()
DiveLabel = np.load()

#Sample Enhancement
#UnderwaterSkel = np.concatenate((UnderwaterSkel, UnderwaterSkel, UnderwaterSkel))
#UnderwaterLabel = np.concatenate((UnderwaterLabel, UnderwaterLabel, UnderwaterLabel))
#DiveSkel = np.concatenate((DiveSkel, DiveSkel))
#DiveLabel = np.concatenate((DiveLabel, DiveLabel))

x_train = np.concatenate((FreeSkel, FlySkel, BackSkel, BreastSkel))
y_train = np.concatenate((FreeLabel, FlyLabel, BackLabel, BreastLabel))

#EXPERIMENTAL
#x_train = np.concatenate((FreeSkel, BackSkel))
#y_train = np.concatenate((FreeLabel, BackLabel))

#x_train = np.concatenate((FlySkel, BreastSkel))
#y_train = np.concatenate((FlyLabel, BreastLabel))

print(x_train.shape)
print(y_train.shape)


def inject_noise(coordinates, mean=0, std=0.3):
    noise = np.random.normal(mean, std, coordinates.shape)
    noisy_coordinates = coordinates + noise
    return noisy_coordinates

tx_train = x_train.copy()
ty_train = y_train.copy()

def noiseAugment(m):
    global x_train, y_train

    for _ in range(m):
        noisySkel = inject_noise(tx_train, mean=0, std=0.5)
        noisyLabel = ty_train
        x_train = np.concatenate((x_train, noisySkel))
        
        y_train = np.concatenate((y_train, ty_train))
        

def rotateAugment(coordinates, max_angle=45):
    angles = np.random.uniform(low=-max_angle, high=max_angle)
    center = np.mean(coordinates, axis=(0, 1))  # Compute the center of rotation

    rotation_matrix = np.array([[np.cos(np.radians(angles)), -np.sin(np.radians(angles))],
                                [np.sin(np.radians(angles)), np.cos(np.radians(angles))]])

    rotated_coordinates = np.zeros_like(coordinates)

    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            # Translate coordinates to the center of rotation
            translated_coord = coordinates[i, j] - center

            # Apply rotation to the translated coordinates
            rotated_coord = np.dot(rotation_matrix, translated_coord.T).T

            # Translate back to the original position
            rotated_coordinates[i, j] = rotated_coord + center

    return rotated_coordinates

def mirrorAugment():
    global x_train, y_train, tx_train, ty_train
    
    fd = tx_train.copy()
    fd[:,:,:,0] = -fd[:,:,:,0]
    x_train = np.concatenate((x_train, fd), axis=0)
    y_train = np.concatenate((y_train, ty_train))
    tx_train = x_train.copy()
    ty_train = y_train.copy()


#7
mirrorAugment()
noiseAugment(2)


print(x_train.shape)
print(y_train.shape)
print(ty_train.shape)


y_test = np.load('y2_test.npy')
x_test = np.load('x2_test.npy')

tx_test = x_test.copy()
ty_test = y_test.copy()
# Generate the same permutation indices

def shuffleTrain(n):
    global x_train, y_train
    for _ in range(n):
        permutation_indices = np.random.permutation(len(x_train))
        x_train = x_train[permutation_indices]
        y_train = y_train[permutation_indices]
        
def shuffleTest(n):
    global x_test, y_test
    for _ in range(n):
        permutation_indices = np.random.permutation(len(x_test))
        x_test = x_test[permutation_indices]
        y_test = y_test[permutation_indices]

# Shuffle both arrays using the same permutation indices

def noiseAugment(m):
    global x_test, y_test

    for _ in range(m):
        noisySkel = inject_noise(tx_test, mean=0, std=.5)
        noisyLabel = ty_test
        x_test = np.concatenate((x_test, noisySkel))
        y_test = np.concatenate((y_test, noisyLabel))
        


print(x_test.shape)
print(y_test.shape)

#EXPERIMENTAL
#noiseAugment(5)

print(x_test.shape)
print(y_test.shape)


i=0
sx_test = []
sy_test = []
"""
Originals:
0-Free
1-Fly
2-Back
3-Breast
4-Underwater
5-Dive
New:
0-Free
1-Back
2-Fly
3-Breast
4-Underwater
5-Dive
"""
while i<y_test.shape[0]:
    if y_test[i] == 0 or y_test[i] == 1 or y_test[i] == 2 or y_test[i] == 3:
        sx_test.append(x_test[i])
        sy_test.append(y_test[i])
    i +=1
"""while i<y_test.shape[0]:
    if y_test[i] == 0 or y_test[i] == 1:
        sx_test.append(x_test[i])
        sy_test.append(0)
    elif y_test[i] == 2 or y_test[i] == 3:
        sx_test.append(x_test[i])
        sy_test.append(1)
    i +=1"""
sx_test = np.array(sx_test)
sy_test = np.array(sy_test)
"""sy_test = np.reshape((sy_test.flatten() - 2), (sy_test.shape[0], 1))
y_train = np.reshape((y_train.flatten() - 2), (y_train.shape[0], 1))"""

print(sx_test.shape)
print(sy_test.shape)



shuffleTrain(1)
shuffleTest(1)


"""x_train_reshaped = x_train.reshape(-1, 32*12*2)
x_test_reshaped = sx_test.reshape(-1, 32*12*2)
x_train = preprocessing.normalize(x_train_reshaped)
sx_test = preprocessing.normalize(x_test_reshaped)

# Reshape back to original shape
x_train = x_train.reshape(-1, 32, 12, 2)
sx_test = sx_test.reshape(-1, 32, 12, 2)
"""

a = x_train.shape[0]
x_train = np.reshape(np.column_stack((np.reshape(x_train, (a*32*12, 2)), np.tile(np.array([0,1,0,1,0,1,0,1,0,1,0,1]), (a*32)))), (a,32,12,3))
b = sx_test.shape[0]
sx_test = np.reshape(np.column_stack((np.reshape(sx_test, (b*32*12, 2)), np.tile(np.array([0,1,0,1,0,1,0,1,0,1,0,1]), (b*32)))), (b,32,12,3))

print(f'X Train Shape: {x_train.shape}')
print(f'Y Train Shape: {y_train.shape}')
print(f'X Test Shape: {sx_test.shape}')
print(f'Y Test Shape: {sy_test.shape}')
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x2_test.npy', sx_test)
np.save('y2_test.npy', sy_test)

#print(x_train[0])
#print(x_test[0])
c=0
for i in sy_test:
    if i == 1:
        c+=1
print(c)
#print(y_test)
c=0
for i in y_train:
    if i == 0:
        c+=1
print(c)
#print(np.sum((x_train > 0.1) & (x_train < 1)))
#print(x_test[0])
#print(y_train)
