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

 path = "front_face.xml" 
faceCascade = cv2.CascadeClassifier(path)
color = (255, 0, 0)
cap = cv2.VideoCapture(0) # capture video using webcam
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = cv2.resize(roi_color, (224, 224))
        roi_color = np.expand_dims(roi_color, axis=0)
        result = model.predict(roi_color)
        ans = result.argmax(axis=-1)
        top_left = (x, y)
        bottom_right = (x+w, y+h)
        cv2.rectangle(img,top_left,bottom_right,color, 3)
        to_write = names[ans[0]]
        cv2.putText(img,to_write,(x+10,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        print("Prediction = " + names[ans[0]])
        cv2.imwrite('Prediction_WebCam.jpg', img)
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()