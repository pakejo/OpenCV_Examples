import cv2

def detect():
    face_cascade = cv2.CascadeClassifier(
        './haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x,y) ,(x+w, y+h), (255, 0, 0), 2)
            eyes = eye_cascade.detectMultiScale(gray, 1.03, 5, 0, (40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow("camera", frame)

        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
