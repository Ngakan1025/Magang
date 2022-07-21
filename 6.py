# Importing libraries
import cv2
import numpy as np

# menggunakan algoritma LBPH dari library opencv
recognition = cv2.face.LBPHFaceRecognizer_create()

# membaca file model train yang sudah dilakukan
recognition.read('train/trainer70.yml')
# Face and eye cascade classifiers from xml files

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX  # font
id = " "  # set default id dengan string kosong


# menampung nama yang akan di recognisi ke dalam suatu list arrayq
nama = ['Tidak Diketahui', 'Soma', 'Elon', 'Krisna']


first_read = True  # set variable first read  untuk membaca mata dengan True

cam = cv2.VideoCapture(0)  # membuka kamera dengan webcam bawaan laptop
cam.set(3, 680)  # set lebar windows
cam.set(4, 480)  # set tinggi win

weightMin = 0.1*cam.get(3)  # weightmin
heightMin = 0.1*cam.get(4)  # heightmin

retV, frame = cam.read()  # mengambil frame dari camera dan ditampilkan


while retV:  # ini akan membuat webcam terus berjalan dan akan menangkap frame setiap perulangan while

    retV, frame = cam.read()

    # convert frame RGB menjadi BG gray
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # menggunakan bilateral filter untuk menghapus kotoran gambar
    abuAbu = cv2.bilateralFilter(abuAbu, 5, 1, 1)

    wajah = faceCascade.detectMultiScale(
        abuAbu, 1.2, 5, minSize=(int(weightMin), int(heightMin)))  # untuk mendeteksi wajah

    if len(wajah) > 0:  # jika jumlah wajah lebih dari 0
        for (x, y, w, h) in wajah:  # membuat perulangan true dengan parameter x,y titik temu dan w = width(lebar) dan h= height(tinggi) dari gambar

            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # membuat kotak yang akan mendetect wajah

            roi_face = abuAbu[y:y + h, x:x + w]  # face detector
            # frame
            roi_face_clr = frame[y:y + h, x:x + w]  # frame dngn bg RGB

            # untuk deteksi mata
            eyes = eyeCascade.detectMultiScale(
                roi_face, 1.3, 5, minSize=(50, 50))

            for (ex, ey, ew, eh) in eyes:  # membuat perulangan dengan parameter ex,ey titik temu dan ew = width(lebar) dan eh= height(tinggi) dari gambar
                cv2.rectangle(roi_face_clr, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)  # membuat kotak yang akan mendetect wajah
                if len(eyes) >= 2:  # jika jumlah mata lebih dari 2
                    if first_read:  # maka jika jumlah mata lebih dari 2 adalah true
                        cv2.putText(frame, "Mata terdeteksi, kedipkan mata untuk pengenalan wajah", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 255, 0), 2)  # maka di print
                    else:  # selain itu
                        cv2.putText(frame, "Mata terbuka", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 255, 255), 2)  # maka di print

                else:  # selain itu
                    if first_read:  # jika first_read nya bukan True maka
                        cv2.putText(frame, "Mata tidak terdeteksi", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (255, 0, 255), 2)  # maka di print

                    else:  # selain itu jika mata berkedip
                        cv2.putText(frame, "Kedipan Mata Terdeteksi !!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                    1, (0, 0, 0), 2)  # maka di print

                        id, confidence = recognition.predict(
                            abuAbu[y:y+h, x:x+w])
                        """menggunakan variabel id berdasarkan id yg 
                        direkam dan confidance untuk mempredict dari file train"""

                        if (confidence > 60):  # jika nilai confidence kurang dari 69 ,bagusnya <60
                            # maka digunakan variable nama sesuai id yang telah dibuat
                            id = nama[id]

                        else:  # selain itu
                            # maka digunakan variable nama dengan index pertama adalah 0
                            id = nama[0]

        # menampilkan nama sesuai id dari wajah yang terekam
        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.waitKey(10)
        # print("Blink Detected.....!!!!")
    else:  # selain itu
        cv2.putText(frame, "Tidak Terdeteksi Wajah.", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 255, 255), 2)  # maka di print
    cv2.imshow('Camera', frame)  # menampilkan window
    # tekan q atau esc untuk keluar
    # ord(ch) returns the ascii of ch
    k = cv2.waitKey(10)
    if k == 27 or k == ord('q'):
        break
    else:
        first_read = False

# release the web-cam
cam.release()
# close the window
cv2.destroyAllWindows()
