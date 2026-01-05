import cv2
import numpy as np

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Load Haar cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
blur_faces = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # DNN face detection
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    output = frame.copy()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if blur_faces:
                face_roi = output[y1:y2, x1:x2]
                face_roi = cv2.blur(face_roi, (40, 15))
                output[y1:y2, x1:x2] = face_roi
            
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)


            # Put confidence as a percentage above the rectangle
            text = f"{confidence:10f}"
            cv2.putText(output, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #EYE DETECTION INSIDE FACE
            roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                pad = int(0.4 * ew)
                pady = int(0.2 * ew)
                #cv2.rectangle(output[y1 + ey:y1 + ey + eh, x1 + ex:x1 + ex + ew].copy(),
                             # (0, 0), (0, 0), (0, 255, 0), 2)  #tf is this for?
                cv2.rectangle(output, (x1 + ex + pad, y1 + ey + pad), (x1 + ex + ew - pad, y1 + ey + eh - pad), (0, 255, 0), 2)
                #cv2.circle(output, (x1 + ex + ew//2, y1 + ey + eh//2), min(ew, eh)//2, (255, 0, 0), 2)
                cv2.rectangle(output, (x1 + ex, y1 + ey), (x1 + ex + ew, y1 + ey + eh), (255, 0, 0), 2)

    cv2.imshow("DNN Face Detection", output)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("b"):
        blur_faces = not blur_faces  # toggle blurring

cap.release()
cv2.destroyAllWindows()
