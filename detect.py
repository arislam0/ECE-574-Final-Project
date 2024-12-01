import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
# The mixer module is useful for projects requiring background music,
# sound effects, or dynamic audio control.
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haarcascade_righteye_2splits.xml')
# These classifiers can work together
# to identify and localize the face and both eyes within an image.
lbl = ['Close', 'Open']

# potentially for saving files or accessing resources relative to this directory.


model = load_model('models/eyes.h5')


path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# complex font style from OpenCV’s
# font options for use in overlaying text on video frames.


count = 0
score = 0
ff = 2
rpred = [99]
lpred = [99]

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Captures a frame from the
    # video feed (ret is a success flag, and frame holds the image data).
    # Converts the frame from color (BGR) to grayscale.

    faces = face.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height),
                  (0, 0, 0), thickness=cv2.FILLED)
    # The image or video frame, The top-left corner, he bottom-right corner,  BGR format,
    # in this case black, rectangle completely with the specified color.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)

        '''
        Converts the reye image from color (BGR) to grayscale,
        Resizes the reye image to 24x24 pixels,  
        Normalizes the pixel values to the range [0, 1] by dividing by 255,
        Reshapes the reye image to a 3D shape (24, 24, 1,),
        np.argmax retrieves the index of the highest predicted probability, 
        providing the predicted class for the right eye (rpred).
        '''
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        # sound.play()
        score = score-1
        cv2.putText(frame, "Open", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:'+str(score), (100, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 15):
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if (ff < 16):
            ff = ff+2
        else:
            ff = ff-2
            if (ff < 2):
                ff = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), ff)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
