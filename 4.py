from tracemalloc import stop
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from scipy.spatial import distance as dist

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()

# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')

# melakukan load classifier haarcascade_facefrontal_default.xml
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_TRIPLEX  # font
id = " "  # set default id dengan string kosong

# menampung nama yang akan di recognisi ke dalam suatu list arrayq
nama = ['Unknown', 'Soma', 'Elon', 'Krisna']

cam = cv2.VideoCapture(0)  # membuka kamera

cam.set(3, 680)  # lebar windows
cam.set(4, 480)  # tinggi windows

weightMin = 0.1*cam.get(3)
heightMin = 0.1*cam.get(4)

# untuk mengetahui orang menutup mata atau tidak


def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])  # horizontal
    B = dist.euclidean(eye[2], eye[4])  # horizontal
    C = dist.euclidean(eye[0], eye[3])  # vertikal

    EAR = (A + B) / (2.0 * C)  # rumus perhitungan mata tertutup atau terbuka

    return EAR  # untuk memanggil kembali


# indeks mata kiri dan kanan
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

counter = 0  # setiap tutup mata maka akan bertambah 1
eyes_ear = 0.2  # batas orang menutup mata atau tidak

detector = dlib.get_frontal_face_detector()  # variabel untuk menargetkan muka
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # untuk memuat sebuah model

while True:  # ini akan membuat webcam terus berjalan dan akan menangkap frame setiap perulangan while

    _, frame = cam.read()  # mengambil frame dari camera dan ditampilkan

    if frame is None:
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = imutils.resize(frame, width=500)

    # convert frame RGB menjadi BG gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # untuk mendeteksi wajah
    wajah = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(weightMin), int(heightMin))
    )

    faces = detector(gray)

    # membuat perulangan true dengan parameter x,y titik temu dan w = width(lebar) dan h= height(tinggi) dari gambar
    for(x, y, w, h) in wajah:

        for (i, face) in enumerate(faces):

            point = predictor(gray, face)
            # mengubah bentuk koorfdinat dari point menjadi bentuk numpy array
            points = face_utils.shape_to_np(point)

            # mengubah bentuk koorfdinat mata kiri menjadi bentuk numpy array
            leftEye = points[lStart:lEnd]
            # mengubah bentuk koorfdinat mata kanan menjadi bentuk numpy array
            rightEye = points[rStart:rEnd]

            # memanggil fungsi calculate_EAR untuk melakukan perhitungan EAR
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)

            # perhitungan mencari nilai EAR
            EAR = (leftEAR + rightEAR) / 2.0

            # menampilkan visual untuk garis pada mata
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            # jika nilai EAR lebih kecil dari eyes_ear maka akan di print seperti dibawah
            if EAR < eyes_ear:
                # menggunakan variabel id berdasarkan id yg direkam dan confidance untuk mempredict dari file train
                id, confidence = recognition.predict(gray[y:y+h, x:x+w])

                # mengecek jika confidence kurang dari 100 ==> "0" berarti sempurna
                if (confidence <= 70):
                    id = nama[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = nama[0]
                    confidence = "  {0}%".format(round(100 - confidence))

            # jika EAR lebih besar maka counter akan tetap 0 atau tidak berubah
            else:
                counter = 0

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)
            # cv2.putText(frame, str(confidence), (x+5, y+h-5),
            #             font, 1, (255, 255, 0), 1)

            cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows
