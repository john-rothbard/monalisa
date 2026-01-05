import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()

#FACE AND EYE TRACKING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #(color, scalefactor (higher = more efficient, worse), minneighbors (faces it detects))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #returns the location of all faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 0, 255), 10)
        #find eyes (will be inside face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            #(#, #, #) color of eye rectangle
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 100, 100), 5)

#COLOR STUFF
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 5, 120])
    upper_yellow = np.array([40, 50, 255])
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])
    yellowmask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    whitemask = cv2.inRange(hsv, lower_white, upper_white)


    result = cv2.bitwise_and(frame, frame, mask=yellowmask)
    #bitwise and mask: 1 1 = 1, 0 1 = 0, 0 0 = 0
    output = frame.copy()
    output[yellowmask > 0] = (0, 0, 255)  # BGR: red
    result = cv2.bitwise_and(frame, frame, mask=whitemask)
    output[whitemask > 0] = (255, 100, 100)

    cv2.imshow('frame', cv2.flip(output,1))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()