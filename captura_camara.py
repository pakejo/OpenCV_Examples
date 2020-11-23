import cv2

clicked = False

def onMouse(event, x, y, flags, param):
    global clicked

    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0) #Inicia la cámara
cv2.namedWindow('Ventana')  #Crea una ventana
cv2.setMouseCallback('Ventana',onMouse) #Añade un evento a la camara 

print ('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read() #Lee un frame 

#Vamos leyendo frame
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('Ventana', frame)    #Muestra el frame en la ventana
    success, frame = cameraCapture.read()  

cv2.destroyWindow('Ventana') #Cierra la ventana
cameraCapture.release() #Libera la camara