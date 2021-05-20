import tensorflow
import cv2
import numpy as np

names = ['bhuvneshwar_kumar',
 'dinesh_karthik',
 'hardik_pandya',
 'jasprit_bumrah',
 'k._l._rahul',
 'kedar_jadhav',
 'kuldeep_yadav',
 'mohammed_shami',
 'ms_dhoni',
 'ravindra_jadeja',
 'rohit_sharma',
 'shikhar_dhawan',
 'vijay_shankar',
 'virat_kohli',
 'yuzvendra_chahal']

model = tensorflow.keras.models.load_model('cric.h5')

path = input("Enter path of image to be predicted: ")

img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
result = model.predict(img)
ans = result.argmax(axis=-1)

print("Prediction: " + names[ans[0]])