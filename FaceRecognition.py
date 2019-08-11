# importing cv2 library
import cv2
# loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')
# face detetcting function
def detect(grey, frame):
    faces = face_cascade.detectMultiScale(grey,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grey[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles = smile.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=3)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame
# Doing face recognition with webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()