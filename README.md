# Indian-Cricketers-Face-Recognition
This repository contains an implementation of VGGFace neural network model to classify the images of Indian Cricketers

# About Data Set
- Data Set was downloaded from https://www.kaggle.com/omkarjc27/indian-cricketers-images
- Contains images of 15 indian cricket players

# Specifications:
- Training Data Set Size = 313 images belonging to 15 classes.
- Testing Data Set size = 78 images belonging to 15 classes.
- Training Acuracy = 99%
- Testing Accuracy = 93.59%

# Dependencies
- Keras
- OpenCV
- NumPy

# Downloads
- Download the trained model cric.h5 from https://drive.google.com/file/d/1T1fLmEA0Hvi78cYlyazHOkAo2RdAQ2Li/view?usp=sharing
- Download Original and Pre Processed datset available at https://drive.google.com/drive/folders/1iUpiqq9MBdynJH3LCUruNW-OoTArIAuB?usp=sharing

# How to Run (Classify which cricketer is in the given image)
1. Download the trained model from the link given above 
2. Run predict.py
3. Enter the path of the image you want to predict

![Screenshot_2](https://user-images.githubusercontent.com/43947335/118996224-5ca52000-b9a5-11eb-8684-b9278f3f8eb1.jpg)
![Predicted_Image](https://user-images.githubusercontent.com/43947335/118995528-d7ba0680-b9a4-11eb-8ea0-ce2d2282723c.jpg)

# How to Run (Open WebCam make a bounding box around face and predict)
1. Download the trained model from the link given above 
2. Run WebCam.py
3. This will open the webcam and create a bounding box around the face and predict on that face

![Prediction_WebCam](https://user-images.githubusercontent.com/43947335/118989711-e18d3b00-b99f-11eb-829f-963e3146a0ce.jpg)

# Dataset Preprocessing
- The dataset was preprocessed to create separate Train and Test folders 80% Train and 20% test.
- Implementation in Dataset Pre Processing.ipynb
