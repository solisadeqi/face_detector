import cv2

# trained face data
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# turn on cam
webcam=cv2.VideoCapture(0)


while True:
    fraem_read , frame = webcam.read()

    gray_cam=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face_coordinates=trained_face_data.detectMultiScale(gray_cam)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(2,255,3),4)

    cv2.imshow('face detector',frame)
    cv2.waitKey(1)

     
webcam.release()

print('finished!')