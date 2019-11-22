import cv2
import numpy as np
from matplotlib import pyplot as plt


# Facial detection/extraction
img = cv2.imread('images/real/real_full_frame_1.png', 0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = faceCascade.detectMultiScale(
    img,
    scaleFactor=2
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
    #lastimg = cv2.resize(faceimg, (32, 32))
    i += 1
    cv2.imwrite("image%d.png" % i, faceimg)

#cv2.imshow("Faces found", img)
#cv2.waitKey(0)

#FFT algorithm

f = np.fft.fft2(faceimg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))

plt.figure()
plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()