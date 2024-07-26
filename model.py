#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import time as time
import os
import numpy as np
import albumentations as A


# In[2]:


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection=mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# In[3]:


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2), 
    A.RGBShift(p=0.2), 
    A.VerticalFlip(p=0.5)
])


# In[4]:


def detect_face(img):
    imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=face_detection.process(imgrgb)
    if results.detections:
        for detection in results.detections:
            data=detection.location_data.relative_bounding_box
            h,w,c=img.shape
            x1,y1,x2,y2=int(data.xmin*w),int(data.ymin*h),int((data.xmin+data.width)*w),int((data.ymin+data.height)*h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = img[y1:y2, x1:x2]
            if roi.size != 0:
                small = cv.resize(roi, (128, 128))
                return small
                
            


# In[5]:


X=[]
y=[]
for img_path in os.listdir("MEN"):
    full_path=os.path.join('MEN',img_path)
    img=cv.imread(full_path)
    smol=detect_face(img)
    if smol is not None:
        X.append(smol)
        y.append(0)
    for i in range (10):
        augmented = transform(image=img)
        augmented_image = augmented['image']
        smol=detect_face(augmented_image)
        if smol is not None:
            X.append(smol)
            y.append(0)
        
for img_path in os.listdir("WOMAN"):
    full_path=os.path.join('WOMAN',img_path)
    img=cv.imread(full_path)
    if img is None:
            continue
    smol=detect_face(img)
    if smol is not None:
        X.append(smol)
        y.append(1)
    for i in range (10):
        augmented = transform(image=img)
        augmented_image = augmented['image']
        smol=detect_face(augmented_image)
        if smol is not None:
            X.append(smol)
            y.append(1)
X=np.array(X)
y=np.array(y)
print(X.shape)
print(y.shape)


# In[6]:


counts = np.bincount(y)
print("Count of 0s:", counts[0])
print("Count of 1s:", counts[1])


# In[7]:


import matplotlib.pyplot as plt
idx=4069
plt.imshow(X[idx])
plt.title(y[idx])
plt.show()


# In[8]:


X = X.astype('float32') / 255.0


# In[9]:


shuffle_indices = np.random.permutation(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set - X:", X_train.shape, "y:", y_train.shape)
print("Testing set - X:", X_test.shape, "y:", y_test.shape)


# In[11]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape # type: ignore
from tensorflow.keras.models import Sequential 


# In[12]:


model = Sequential([
    Conv2D(filters=16, kernel_size=(2, 2),strides=1, activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(2, 2),strides=1,  activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(2, 2),strides=1, activation='relu' ),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=256, activation='relu'),
    Dense(units=1, activation='sigmoid')
])


# In[13]:


model.summary()


# In[14]:


t0 = time.time()
from tensorflow.keras.losses import binary_crossentropy
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
t1 = time.time()
total = t1-t0
print(total)


# In[ ]:


model.save('gender_recogniser.h5')


# In[ ]:


path='WOMAN/0029.jpg'
img=cv.imread(path)
imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
results=face_detection.process(imgrgb)
if results.detections:
    for detection in results.detections:
        data=detection.location_data.relative_bounding_box
        h,w,c=img.shape
        x1,y1,x2,y2=int(data.xmin*w),int(data.ymin*h),int((data.xmin+data.width)*w),int((data.ymin+data.height)*h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size != 0:
            small = cv.resize(roi, (128, 128))
            small = small.astype('float32') / 255.0
            small=small.reshape(1,128,128,3)
            ypred=model.predict(small)
            plt.imshow(imgrgb) 
            if(ypred>0.5):
                plt.title('WOMAN')
            else:
                plt.title('MAN')
else:
    print("Face not visible clearly :(")
plt.show()


