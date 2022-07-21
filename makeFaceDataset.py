import cv2
import os
cam = cv2.VideoCapture(0)  # membuka camera
cam.set(3, 648)  # ubah lebar video
cam.set(4, 348)  # ubah tinggi video
# menggunakan model xml haarcascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# memasukan nomor dari dataset yang akan direkam
faceId = input(
    "Masukan nama data wajah yang akan direkam dan Tetapkan wajah anda didepan kamera laptop.[kemudian tekan ENTER] :")
print(" Tunggu proses pengambilan data wajah selesai ...")

ambilData = 0  # membuat variable yang menampung jumlah data bg gray

# merekam dataset gambar format bg grayscale
while True:  # selama true akan melooping dan camera akan menyala
    revT, frame = cam.read()  # mengambil frame dari camera dan ditampilkan
    # variabel untuk membuat gambar bg menjadi gray
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame ,scalefactor, minNeighboar
    wajah = faceCascade.detectMultiScale(abuAbu, 1.3, 5)

    for(x, y, w, h) in wajah:  # membuat perulangan true dengan parameter x,y titik temu dan w = width(lebar) dan h= height(tinggi) dari gambar
        # membuat kotak yang akan mendetect wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ambilData += 1  # mengambil 1 data gambar tiap frame

        # menyimpan data wajah ke dalam folder dataset
        # membaca data yang telah diambil lalu disimpan ke dalam folder
        cv2.imwrite("datasetwajah/user." + str(faceId) + '.' +
                    str(ambilData) + ".jpg", abuAbu[y:y+h, x:x+w])

    # menampilkan camera serta gambar yg dihasilkan kamera
    cv2.imshow('Camera', frame)

    k = cv2.waitKey(100) & 0xff  # menunngu 100 milisecond per frame yg diambil
    # jika menekan tombol escape dan q maka camera ditutup serta perekaman akan selesai
    if k == 27 or k == ord('q'):
        break
    elif ambilData >= 50:  # jika data sudah diambil sebanyak 50 maka akan perekaman akan berhenti
        break

# ambilData = 0  #membuat variable yang menampung jumlah data bg gray

# #merekam dataset gambar mata pada dataset
# while True: # selama true akan melooping dan camera akan menyala
#     revT, frame = cam.read() #mengambil frame dari camera dan ditampilkan
#     abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # variabel untuk membuat gambar bg menjadi gray
#     mata = eyeCascade.detectMultiScale(abuAbu, 1.1, 5) #frame ,scalefactor, minNeighboar

#     for(x, y , w, h) in mata: #membuat perulangan true dengan parameter x,y titik temu dan w = width(lebar) dan h= height(tinggi) dari gambar
#         cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2) #membuat kotak yang akan mendetect wajah
#         ambilData += 1 #mengambil 1 data gambar tiap frame

#         #menyimpan data wajah ke dalam folder dataset
#         cv2.imwrite("datasetwajah/mata/user." + str(faceId) + '.' + str(ambilData) + ".jpg", abuAbu[y:y+h,x:x+w]) #membaca data yang telah diambil lalu disimpan ke dalam folder

#     cv2.imshow('Camera', frame) #menampilkan camera serta gambar yg dihasilkan kamera

#     k = cv2.waitKey(100) & 0xff # menunngu 100 milisecond per frame yg diambil
#     if k == 27 or k == ord('q'):  #jika menekan tombol escape dan q maka camera ditutup serta perekaman akan selesai
#         break
#     elif ambilData >= 50: #jika data sudah diambil sebanyak 50 maka akan perekaman akan berhenti
#         break

ambilData = 0  # membuat variable yang menampung jumlah data bg color

while True:
    revT, frame = cam.read()
    abuAbu = frame
    # frame ,scalefactor, minNeighboar
    wajah = faceCascade.detectMultiScale(abuAbu, 1.3, 5)

    for(x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ambilData += 1

        # menyimpan data wajah ke dalam folder dataset
        cv2.imwrite("datatestwajah/user." + str(faceId) + '.' +
                    str(ambilData) + ".jpg", abuAbu[y:y+h, x:x+w])

    cv2.imshow('Camera', frame)
    k = cv2.waitKey(100) & 0xff
    if k == 27 or k == ord('q'):
        break
    elif ambilData >= 30:  # mengambil 30 dataset format RGB
        break

# ambilData = 0 #membuat variable yang menampung jumlah data mata pada bg color

# while True:
#     revT, frame = cam.read()
#     abuAbu = frame
#     mata = eyeCascade.detectMultiScale(abuAbu, 1.1, 5) #frame ,scalefactor, minNeighboar

#     for(x, y , w, h) in mata:
#         cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
#         ambilData += 1

#         #menyimpan data wajah ke dalam folder dataset
#         cv2.imwrite("datatestwajah/mata/user." + str(faceId) + '.' + str(ambilData) + ".jpg", abuAbu[y:y+h,x:x+w])

#     cv2.imshow('Camera', frame)
#     k = cv2.waitKey(100) & 0xff
#     if k == 27 or k == ord('q'):
#         break
#     elif ambilData >= 30: #mengambil 30 dataset format RGB
#         break

print("Pengambilan Gambar selesai")
cam.release()
cv2.destroyAllWindows()
