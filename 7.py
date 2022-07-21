
import cv2
from scipy.spatial import distance as dist
import numpy as np
import dlib
import imutils
from imutils import face_utils
from datetime import datetime

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()
# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')
# Face and eye cascade classifiers from xml files
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" 
dlib describes eye with 6 points.
when you blink, the EAR value will change from 0.3 to  near 0.05
"""

# if you have glasses, you should cahnge the threshold!
Eye_AR_Thresh = 0.3

# how many frames shows the blink( use it for reduce noises)
Eye_AR_Consec_frames = 3

# initialize the frame counters and the total number of blinks
counter = 0
total = 0

font = cv2.FONT_HERSHEY_COMPLEX
id = " "  # set default id dengan string kosong

# menampung nama yang akan di recognisi ke dalam suatu list array
nama = ['Tidak Diketahui', 'Krisna', 'Risma', 'Soma']


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A+B) / (2*C)

    # return the eye aspect ratio
    return ear


def markAttendance(name):
    with open("datadetect.csv", 'r+') as f:
        namesDatalist = f.readlines()
        namelist = []
        for line in namesDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)
weightMin = 0.1*cam.get(3)  # weightmin
heightMin = 0.1*cam.get(4)  # heightmin

while True:

    ret, frame = cam.read()
    if frame is None:
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    wajah = faceCascade.detectMultiScale(
        gray, 1.2, 5, minSize=(int(weightMin), int(heightMin)))

    for(x, y, w, h) in wajah:

        # loop over the face detections
        for face in rects:
            (x1, y1) = (face.left(), face.top())
            (x2, y2) = (face.right(), face.bottom())

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # find left and right eyes and calculate the EAR
            # left eye is 37-42th points (numpy starts from 0)
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            EAR = (leftEAR + rightEAR) / 2

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if EAR < Eye_AR_Thresh:
                counter += 1
                id, confidance = recognition.predict(
                    gray[y:y+h, x:x+w])
                """menggunakan variabel id berdasarkan id yg 
                        direkam dan confidance untuk mempredict dari file train"""

                if (confidance < 100):  # jika nilai confidence kurang dari 69 ,bagusnya <60
                    # maka digunakan variable nama sesuai id yang telah dibuat
                    id = nama[id]
                else:  # selain itu
                    # maka digunakan variable nama dengan index pertama adalah 0
                    id = nama[0]

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if counter > Eye_AR_Consec_frames:
                    total += 1

                # reset the eye frame counter
                counter = 0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # cv2.putText(frame, "Blinks: {}".format(total),
            #             (10, 20), font, 0.55, (0, 0, 255), 1)
            # cv2.putText(frame, "EAR: {:.2f}".format(EAR),
            #             (10, 50), font, 0.55, (0, 0, 255), 1)

            cv2.putText(frame, str(id), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == ord('q'):
        break
markAttendance(id)
cam.release()
cv2.destroyAllWindows()
