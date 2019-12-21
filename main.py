import cv2
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt


# Labels for real and fake frames
df = pd.DataFrame({'frame': [], 'label': []})

for i in range(0, 2):
    # Split fake video into frames
    if i == 0:
        vidcap = cv2.VideoCapture('599_585.mp4')
    else:
        vidcap = cv2.VideoCapture('599.mp4')
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        cv2.imwrite("frames/frame{}_{}.jpg".format(i, count), image)
        print ('Read a new frame: ', success)
        count += 1

    # Facial detection/extraction
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for frameNumber in range(0, count):
        img = cv2.imread('frames/frame{}_{}.jpg'.format(i, frameNumber), 0)

        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.85
        )

        print("Found {0} faces!".format(len(faces)))

        # Crop background so that the image is just the face

        j = 0
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            faceimg = cv2.resize(faceimg, (281, 281))
            j += 1
            cv2.imwrite("frames_faces/frame{}_{}_{}.jpg".format(i, frameNumber, j), faceimg)

            #FFT algorithm
            f = np.fft.fft2(faceimg)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 10*np.log(np.abs(fshift))
            frame_file_name = 'frame{}_{}_{}.jpg'.format(i, frameNumber, j)
            df = df.append({'frame': frame_file_name, 'label': i}, ignore_index=True)
            cv2.imwrite('frames_fft/' + frame_file_name, magnitude_spectrum)

        frameNumber = frameNumber + 1

df.to_csv('example.csv')
