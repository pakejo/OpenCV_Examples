import cv2
import csv
import os
import sys
import numpy as np


def create_user_folder(name):
    exists_dir = os.path.exists('./data/' + str(name))

    if not exists_dir:
        parent_dir = "./data"
        path = os.path.join(parent_dir, name)
        os.mkdir(path)


def create_CSV(name):
    file = './data/{}/{}.csv'.format(str(name), str(name))
    path = './data'
    cont = 0

    iter = os.scandir(path)

    for entry in iter:
        if entry.name != str(name):
            cont += 1

    with open(file, 'w') as f:
        writer = csv.writer(f)
        iter = os.scandir('./data/{}/'.format(str(name)))

        for entry in iter:
            writer.writerow([entry.name, cont])


def generate():
    face_cascade = cv2.CascadeClassifier(
        './haarcascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    name = input("Introduce tu nombre: ")

    while (True):
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            create_user_folder(name)
            cv2.imwrite("./data/{}/{}.pgm".format(str(name), str(count)), f)
            count += 1

        cv2.imshow("camera", frame)

        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    create_CSV(name)


def read_images(path, sz=None):
    c = 0
    X, y = [], []

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subjet_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subjet_path):
                try:
                    if filename == ".directory":
                        continue

                    if filename.endswith('csv'):
                        continue

                    filepath = os.path.join(subjet_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    if sz is not None:
                        im = cv2.resize(im, (200, 200))

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)

                except IOError:
                    print("I/O error")
                except:
                    print("Unexpected error")
            c += 1

    return [X, y]


def face_rec():
    names = ['Fran']

    [X, y] = read_images('./data')
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    face_cascade = cv2.CascadeClassifier(
        './haarcascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    while (True):
        _, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                roi = gray[x:x+w, y:y+h]
                params = model.predict(roi)
                cv2.putText(img, names[params[0]], (x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
            cv2.imshow("camera", img)
            
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    #generate()
    face_rec()
