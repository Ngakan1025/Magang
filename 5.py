# Importing libraries
import cv2
import numpy as np

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()

# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')

# Face and eye cascade classifiers from xml files
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

font = cv2.FONT_HERSHEY_TRIPLEX
id = " "

nama = ['No one', 'Soma', 'Elon', 'Krisna']

first_read = True

cam = cv2.VideoCapture(0)  # membuka kamera

cam.set(3, 680)  # lebar windows
cam.set(4, 480)  # tinggi windows

weightMin = 0.1*cam.get(3)
heightMin = 0.1*cam.get(4)


# Video Capturing by using webcam

retTV, image = cam.read()
while retTV:
    # this will keep the web-cam running and capturing the image for every loop
    retTV, image = cam.read()

    # Convert the rgb image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying bilateral filters to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # to detect face
    faces = faceCascade.detectMultiScale(
        gray, 1.2, 5, minSize=(int(weightMin), int(heightMin)))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), (1, 190, 200), 2)
            # face detector
            roi_face = gray[y:y + h, x:x + w]
            # image
            roi_face_clr = image[y:y + h, x:x + w]
            # to detect eyes
            eyes = eyeCascade.detectMultiScale(
                roi_face, 1.3, 5, minSize=(50, 50))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_face_clr, (ex, ey),
                              (ex+ew, ey+eh), (255, 153, 255), 2)
                if len(eyes) >= 2:
                    if first_read:
                        cv2.putText(image, "Eye's detected, press s to check blink", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 0), 2)
                    else:
                        cv2.putText(image, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 255, 255), 2)

                else:
                    if first_read:
                        cv2.putText(image, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 255), 2)
                    else:
                        cv2.putText(image, "Kedipan Mata Terdeteksi !!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (0, 0, 0), 2)  # maka di print
                        id, confidence = recognition.predict(
                            gray[y:y+h, x:x+w])

                        # mengecek jika confidence kurang dari 100 ==> "0" berarti sempurna
                        if (confidence > 60):
                            id = nama[id]
                            confidence = "  {0}%".format(
                                round(100 - confidence))
                        else:
                            id = nama[0]
                            confidence = "  {0}%".format(
                                round(100 - confidence))

                        cv2.putText(image, str(id), (x+5, y-5),
                                    font, 1, (255, 255, 255), 2)
                        cv2.putText(image, str(confidence), (x+5, y+h-5),
                                    font, 1, (255, 255, 0), 1)

    else:
        cv2.putText(image, "No Face Detected.", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 255, 255), 2)

    cv2.imshow('Camera', image)
    # a = cv2.waitKey(1)
    # # press q to Quit and S to start
    # # ord(ch) returns the ascii of ch
    # if a == ord('q'):
    #     break
    # elif a == ord('s'):
    #     first_read = False

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    else:
        first_read = False

# release the web-cam
cam.release()
# close the window
cv2.destroyAllWindows()
