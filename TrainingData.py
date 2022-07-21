import cv2
import os
import numpy as np
from PIL import Image

# menentukan path untuk foto yang sudah diambil di dalam folder database
path = 'datasetwajah'
path1 = 'datatestwajah'

recognition = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# fungsi untuk mengambil image serta mengconvert ke array


def getImagesAndLabelsForTrain(path):
    # library os untuk membaca file dari path dataset
    # menjoinkan file dataset ke dalam parameter path yang ada pada fungsi getImages...
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []  # data sample akan disimpan ke dalam list array kosong
    ids = []  # data faceId akan disimpan ke dalam list array kosong

    # mengambil seluruh file  yang telah dibaca kedalam parameter path yang ada di fungsi getImageLabel
    for imagePath in imagePaths:
        # library PILimage untuk membuka file dalam path dan mengconvert ke dalam grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # menyimpan variable PIL_img ke dalam numpy array
        img_numpy = np.array(PIL_img, 'uint8')

        # format file dipisah -1(dari kanan), kita split hilangkan extention .jpg dengan "."
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # mengambil variable img_numpy setelah itu detect menggunakan variabel detector diatas
        faces = detector.detectMultiScale(img_numpy)

        for(x, y, w, h) in faces:  # looping variable faces dengan parameter x,y,w,h
            # menambahkan img_numpy yg sudah di detect ke dalam facesamples
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            # menambahkan id yang sudah di split atau menghilangkan extention jpg ke dalam variable ids
            ids.append(id)

    return faceSamples, ids  # mengembalikan facesamples dan id


print("\n [INFO] Proses Training. ini akan memakan waktu beberapa detik, mohon menunggu... ")

# proses training
# faces dan ids akan disimpan ke dalam path dataset
faces, ids = getImagesAndLabelsForTrain(path)
# mentraining faces dan ids(dlm bntuk numpy array) dengan variable recognition
recognition.train(faces, np.array(ids))

# menyimpan model hasil training ke dalam folder train/trainer.yml
recognition.write('train/trainer70.yml')
recognition.read('train/trainer70.yml')


# mencoba menggunakan datatesting untuk mengukur akurasi dari hasil train model yeng telah dilakukan
totalcount = 0
kumakurasi = []


for i in range(1, 4):
    for j in range(1, 31):
        test_img1 = "datatestwajah/" + "user." + str(i) + "." + str(j) + ".jpg"
        image = Image.open(test_img1).convert('L')
        image_np = np.array(image, 'uint8')
        id, confidence = recognition.predict(image_np)
        totalcount += confidence

    if(j == 30):
        akurasi = 100 - (totalcount/(3*30))
        kumakurasi.append(akurasi)
        totalcount = 0

accmodel = sum(kumakurasi)/i

for i in range(0, 3):
    print("Akurasi model untuk wajah ke " +
          str(i+1)+" adalah : ", kumakurasi[i])

print("Akurasi rata-rata dari model adalah :", accmodel)

# menampilkan jumlah wajah yang dilatih dan program train selesai
print("\n [INFO] {0} wajah telah dilakukan proses training. Keluar dari Program".format(
    len(np.unique(ids)))
)
