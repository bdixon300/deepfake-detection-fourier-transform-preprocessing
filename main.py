import cv2
import os
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def generateFFT(path, label, video_number):
    # Labels for real and fake frames
    df = pd.DataFrame({'frame': [], 'label': []})
    video_count = 0
    for filename in os.listdir(path):
        if video_count > video_number - 1:
            break
        elif video_count < 5:
            print("skipping {} as its in the testing set".format(filename))
            video_count = video_count + 1
            continue
        frame_count = 0
        print("Generating FFT for video: {}".format(filename))
        # Split fake video into frames
        vidcap = cv2.VideoCapture(path + filename)
        success = True
        while success:
            success,image = vidcap.read()
            cv2.imwrite("frames/frame{}_{}_{}.jpg".format(video_count, frame_count, label), image)
            #print ('Read a new frame: ', success)
            frame_count += 1

        # Facial detection/extraction
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        for frameNumber in range(0, frame_count):
            img = cv2.imread('frames/frame{}_{}_{}.jpg'.format(video_count, frameNumber, label), 0)

            faces = faceCascade.detectMultiScale(
                img,
                scaleFactor=1.85
            )

            #print("Found {0} faces!".format(len(faces)))

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
                cv2.imwrite("frames_faces/frame{}_{}_{}_{}.jpg".format(video_count, frameNumber, j, label), faceimg)

                #FFT algorithm
                f = np.fft.fft2(faceimg)
                fshift = np.fft.fftshift(f)
                # The magnitude size is likely to be a hyper parameter in this case
                magnitude_spectrum = 10*np.log(np.abs(fshift))
                frame_file_name = 'frame{}_{}_{}_{}.jpg'.format(video_count, frameNumber, j, label)
                df = df.append({'frame': frame_file_name, 'label': label}, ignore_index=True)
                df["label"] = df["label"].astype(int)
                # Create FFT representation of face
                cv2.imwrite('frames_fft/' + frame_file_name, magnitude_spectrum)

        video_count = video_count + 1
        if os.path.isfile('example.csv'):
            print("appending to csv")
            df.to_csv('example.csv', header=None, mode='a')
        else:
            print("Creating new csv")
            df.to_csv('example.csv')

generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/manipulated_sequences/Deepfakes/c23/videos/', 0, 20)
generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/original_sequences/youtube/c23/videos/', 1, 20)

#generateFFT('/Volumes/antideepfake/FaceForensics++dataset/manipulated_sequences/Deepfakes/raw/videos/', 0, 5)
#generateFFT('/Volumes/antideepfake/FaceForensics++dataset/original_sequences/youtube/raw/videos/', 1, 5)

