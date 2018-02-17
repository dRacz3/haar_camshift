import cv2
import sys
import pyautogui

# What fille?
cascPath = 'haar_finger.xml'
cascPath2 = 'haarcascade_frontalface_default.xml'
cascPathSmile = 'haarcascade_smile.xml'
faceCascade = cv2.CascadeClassifier(cascPath2)

video_capture = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 80),
        maxSize = (250, 450),
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
    )

    mouseX, mouseY = pyautogui.position()

    # Draw a rectangle around the faces
    for (x, y, w, h) in results:
        sizeX, sizeY = pyautogui.size()
        height, width, channels = frame.shape
        screenX = (x/height) * sizeX
        screenY = (y/width) *sizeY
        dx = (sizeX-screenX)-mouseX #flip image..
        dy = screenY-mouseY

        kp = 0.8

        pyautogui.moveRel(dx*kp, dy*kp)

#        pyautogui.moveTo(sizeX-x*3,y*3)
        cv2.circle(frame,(int(x + w/2) ,int(y + h/2)), 10, (0,0,255), -1)
#        cv2.circle(frame, (x+w/2, y+h/2))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
