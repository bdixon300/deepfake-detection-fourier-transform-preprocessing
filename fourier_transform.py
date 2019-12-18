import cv2
import numpy as np
from matplotlib import pyplot as plt


# Split video into frames
vidcap = cv2.VideoCapture('599_585.mp4')
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("frames/frame%d.jpg" % count, image)
  print ('Read a new frame: ', success)
  count += 1

# Facial detection/extraction
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for frameNumber in range(0, count):
    img = cv2.imread('frames/frame{}.jpg'.format(frameNumber), 0)

    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.85
    )

    print("Found {0} faces!".format(len(faces)))

    # Crop background so that the image is just the face

    i = 0
    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        #smallimg = cv2.resize(faceimg, (32, 32))
        i += 1
        cv2.imwrite("frames_faces/frame{}_{}.jpg".format(frameNumber, i), faceimg)

        #FFT algorithm
        f = np.fft.fft2(faceimg)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 5*np.log(np.abs(fshift))
        cv2.imwrite("frames_fft/frame{}_{}.jpg".format(frameNumber, i), magnitude_spectrum)

    frameNumber = frameNumber + 1
