from turtle import right
import cv2  # import library opencv

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()

# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')

# melakukan load classifier haarcascade_facefrontal_default.xml
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# rightEyeCascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
# leftEyeCascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")

font = cv2.FONT_HERSHEY_TRIPLEX
id = 0

nama = ['No one', 'Soma', 'Elon', 'Krisna']

cam = cv2.VideoCapture(0)  # membuka kamera

cam.set(3, 680)  # lebar windows
cam.set(4, 480)  # tinggi windows

weightMin = 0.1*cam.get(3)
heightMin = 0.1*cam.get(4)

first_read = True

while True:
    revT, frame = cam.read()  # mengambil frame dari camera dan ditampilkan
    # variabel untuk membuat gambar bg menjadi gray
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    wajah = faceCascade.detectMultiScale(
        abuAbu,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(weightMin), int(heightMin))
    )

    # Eyes detection
    # check first if eyes are open (with glasses taking into account)
    mata = eyeCascade.detectMultiScale(
        abuAbu,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x, y, w, h) in mata:
        roi_gray = abuAbu[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (255, 153, 255), 2)
            if len(eyes) >= 2:
                for(x, y, w, h) in wajah:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    id, confidence = recognition.predict(abuAbu[y:y+h, x:x+w])

                    # mengecek jika confidence kurang dari 100 ==> "0" berarti sempurna
                    if (confidence > 30):
                        id = nama[id]
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        id = nama[0]
                        confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(frame, str(id), (x+5, y-5),
                            font, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(confidence), (x+5, y+h-5),
                            font, 1, (255, 255, 0), 1)
            else:
                if first_read:
                    cv2.putText(frame, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                1, (0, 0, 0), 2)
                    cv2.imshow('image', frame)
                    cv2.waitKey(1)
                    print("Blink Detected.....!!!!")

        # righteye = rightEyeCascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in righteye:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # lefteye = leftEyeCascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in lefteye:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.putText(frame, str(), (x+5, y-5), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', frame)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

print("[INFO] Keluar program")
cam.release()
cv2.destroyAllWindows()
