# -*- coding: utf-8 -*-

#Import the Necessary Libraries
import pickle
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils


#Load the model
model=pickle.load(open('mask_detection_model.pkl', 'rb'))
#We are capturing the faces using cascade classifier which is already present in OpenCV Lib
face_class=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Starting the webcam
source = VideoStream(src=0).start()

#Initializing the labels and it colors respectively. 
label_dict={0:'Mask', 1:'No Mask'}
#since cv2 is of BGR format we are giving the colour respectively 
#Green=255 for Mask and Red = 255 for No Mask
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):
# We are reading the video by converting it into Image streams
    img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#detectMultiScale This will give the x,y and width and height coordinates for our ROI
    faces=face_class.detectMultiScale(gray,1.3,5)  
    #print("[INFO] Predicting....")
    for (x,y,w,h) in faces:
    #Drawing the frame around our face and preprocessing to predict
        face_img = img[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        result=model.predict(reshaped)
    
        label=np.argmax(result,axis=1)[0]
     	
#Drawing the box around our face
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        
        
        cv2.putText(img, label_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    	# show the output frame
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    #Press escape to exit
    if(key==27):
        break
#Cleanup        
cv2.destroyAllWindows()
source.stop()
