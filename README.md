# Indian-Cricketers-Face-Recognition
This repository contains an implementation of VGGFace neural network model to classify the images of Indian Cricketers

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
![Screenshot_1](https://user-images.githubusercontent.com/43947335/118990139-4052b480-b9a0-11eb-9644-cdbf704537d4.jpg)
![Screenshot_10](https://user-images.githubusercontent.com/43947335/118990149-421c7800-b9a0-11eb-9860-f9cdf69f9910.jpg)


# How to Run (Open WebCam make a bounding box around face and predict)
1. Download the trained model from the link given above 
2. Run WebCam.py
3. This will open the webcam and create a bounding box around the face and predict on that face
![Prediction_WebCam](https://user-images.githubusercontent.com/43947335/118989711-e18d3b00-b99f-11eb-829f-963e3146a0ce.jpg)

