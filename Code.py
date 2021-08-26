# DOWNLOADING THE DATASET
#STEP 1
!wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
#STEP 2  
!unzip kagglecatsanddogs_3367a.zip

# IMPORT MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import tqdm
import random
from keras.preprocessing.image import load_img
warnings.filterwarnings('ignore')

# CREATE DATAFRAME FOR INPUT AND OUTPUT
#STEP 1
input_path = []
label = []
for class_name in os.listdir("PetImages"):
  for path in os.listdir("PetImages/"+class_name):
    if class_name == 'Cat':
      label.append(0)
    else:
      label.append(1)
    input_path.append(os.path.join("PetImages",class_name, path))
print(input_path[0], label[0])  

# STEP 2
df = pd.DataFrame()
df['images'] = input_path
df['label'] = label
df = df.sample(frac=1).reset_index(drop = True) ##shuffing the dataframe
df.head()

## seeing the unwanted files in the dataset
for i in df['images']:
  if '.jpg' not in i:
    print(i)

import PIL
l = []
for image in df['images']:
  try:
    img = PIL.Image.open(image)
  except:
    l.append(image)
l

#delete db files
df = df[df['images']!='PetImages/Dog/Thumbs.db']
df = df[df['images']!='PetImages/Cat/Thumbs.db']
df = df[df['images']!='PetImages/Cat/666.jpg']
df = df[df['images']!='PetImages/Dog/11702.jpg']
len(df)

# EXPLORATORY DATA ANALYSIS
##display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label'] == 1]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
  plt.subplot(5,5, index+1)
  img = load_img(file)
  img = np.array(img)
  plt.imshow(img)
  plt.title('Dogs')
  plt.axis('off')
  
##dispaly grid of images
plt.figure(figsize=(25,25))
temp = df[df['label'] == 0]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
  plt.subplot(5,5, index+1)
  img = load_img(file)
  img = np.array(img)
  plt.imshow(img)
  plt.title('Cats')
  plt.axis('off')
  
import seaborn as sns
sns.countplot(df['label'])

# CREATE DATA GENERATOR For THE IMAGES
df['label'] = df['label'].astype('str')
df.head()

#input split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state=42)

from keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(
    rescale = 1./255, ##normalization of images
    rotation_range = 40,   ##augmentation of m=image to avoid overfitting
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_generator = ImageDataGenerator(rescale=1./255)

train_iterator = train_generator.flow_from_dataframe(
    df, 
    x_col = 'images', 
    y_col = 'label', 
    target_size=(128,128),
    batch_size=512,
    class_mode='binary'
)

val_iterator = val_generator.flow_from_dataframe(
    df, 
    x_col = 'images', 
    y_col = 'label', 
    target_size=(128,128),
    batch_size=512,
    class_mode='binary'
)

# MODEL CREATION
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential([
                  Conv2D(16, (3,3), activation = 'relu', input_shape=(128,128,3)),
                  MaxPool2D((2,2)),
                  Conv2D(32, (3,3), activation='relu'),
                  MaxPool2D((2,2)),
                  Conv2D(64, (3,3), activation='relu'),
                  MaxPool2D((2,2)),
                  Flatten(),
                  Dense(512, activation='relu'),
                  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics =['accuracy'])
model.summary()

history = model.fit(train_iterator, epochs=5, validation_data=val_iterator)

# Visualization of Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label= 'validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()
