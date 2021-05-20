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

classifi = 'front_face.xml' # name of file with XML tags
faceCascade = cv2.CascadeClassifier(classifi)

path = input("Enter path of image to be predicted: ")
color = (255, 0, 0)
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
count = 0
for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (224, 224))
        roi_color = np.expand_dims(roi_color, axis=0)
        #print(roi_color.shape)
        result = model.predict(roi_color)
        ans = result.argmax(axis=-1)
        top_left = (x, y)
        bottom_right = (x+w, y+h)
        cv2.rectangle(img,top_left,bottom_right,color, 3)
        to_write = names[ans[0]]
        cv2.putText(img,to_write,(x+10,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        print("Prediction = " + names[ans[0]])
cv2.imwrite("Predicted_Image.jpg", img)
