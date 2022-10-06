import cv2
import numpy as np
import face_recognition
import os
import pickle

path = "images"


images = os.listdir(path)
loadedImages = []
names = []
encodingsList = []

print(images)


# a function to capture image
def captureImage(name):
    # To capture a image using webcam
    cam = cv2.VideoCapture(0)
    result, image = cam.read()

    # to save the image in appropriate location along with the name
    if result == True:
        cv2.imshow("Captured image", image)
        cv2.imwrite(path + '/' + name + '.png', image)
        cv2.waitKey(0)
        cv2.destroyWindow("Captured image")


# To load the images in proper format required to encode
def loadImages():
    images = os.listdir(path)
    for img in images:
        curImg = face_recognition.load_image_file(path + '/' + img)
        loadedImages.append(curImg)
        names.append(img.split(".")[0])


# To encode each image in file and saving the encodings for future use
def encodeImage(name):
    loadImages()
    for img in loadedImages:
        # convert image in RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(face_recognition.face_encodings(img)) > 0:
            encodingsList.append(face_recognition.face_encodings(img)[0])

    encodingDic = {}
    with open('encodings', 'ab') as f:
        print("length of encoding:", len(encodingsList))
        for i in range(0, len(encodingsList)):
            encodingDic[names[i]] = encodingsList[i]
        pickle.dump(encodingDic, f)


def loadEncodings():
    with open('encodings', 'rb') as f:
        encodingDic = pickle.load(f)
        encodingDic = dict(encodingDic)

    for name, encodings in encodingDic.items():
        names.append(name)
        encodingsList.append(encodings)
        

    del encodingDic


# if no images found. We need to train our first image
if len(images) == 0:
    print("Enter your name :- ")
    name = input()
    captureImage(name)
    encodeImage(names)

loadEncodings()
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detecting all faces in curret faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    cv2.imshow('WebCam', img)

    if encodeCurFrame and facesCurFrame:
        matches = face_recognition.compare_faces(encodingsList, encodeCurFrame[0])
        faceDis = face_recognition.face_distance(encodingsList, encodeCurFrame[0])
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = facesCurFrame[0]
        flag = 0

        if (faceDis[matchIndex] > 0.35):
            print('Do you know this person? y/n')

            if cv2.waitKey(0) == ord('y'):
                name = input("Enter name:- ")
                cv2.imwrite(path + '/' + name + '.png', img)
                encodeImage(names)
                loadEncodings()
            
            elif cv2.waitKey(0) == ord('q'):
                break

            elif cv2.waitKey(0) != [ord('y'),ord('q')]:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)


        else:
            if matches[matchIndex]:
                name = names[matchIndex].upper()
                print(name)
                print(faceDis[matchIndex])
                y1, x2, y2, x1 = facesCurFrame[0]

                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                img = cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
                cv2.imshow('WebCam', img)
        

cap.release()
cv2.destroyAllWindows