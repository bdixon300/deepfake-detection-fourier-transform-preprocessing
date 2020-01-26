import cv2
import os
import numpy as np
import pandas as pd 
from PIL import Image
import better_facial_detection as fd
from matplotlib import pyplot as plt

def generateFFT(path, label, video_number):
    # Labels for real and fake frames
    df = pd.DataFrame({'frame': [], 'label': []})
    video_count = 0
    for filename in os.listdir(path):
        # Uncomment to convert a specific video file
        """if filename != "09__walking_down_street_outside_angry.mp4":
            print("skipping {}".format(filename))
            continue"""
        if video_count > video_number - 1:
            break
        # Uncomment to skip video files
        """elif video_count < 6:
            print("skipping {} as its in the testing set".format(filename))
            video_count = video_count + 1
            continue"""
        frame_count = 0
        print("Generating FFT for video: {}".format(filename))
        # Split fake video into frames
        vidcap = cv2.VideoCapture(path + filename)
        success = True
        while success:
            success,image = vidcap.read()
            if not success:
                break
            cropped_faces = fd.MTCNNFacialDetection(Image.fromarray(image), video_count, frame_count, label)
            frame_count += 1

            # Crop background so that the image is just the face
            j = 0
            for face in cropped_faces:
                face = np.array(face)
                face = cv2.resize(face, (281, 281))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                j += 1
                #cv2.imwrite("frames_faces/frame{}_{}_{}_{}.jpg".format(video_count, frame_count, j, label), face)

                #FFT algorithm
                f = np.fft.fft2(face)
                fshift = np.fft.fftshift(f)
                # The magnitude size is likely to be a hyper parameter in this case
                magnitude_spectrum = 10*np.log(np.abs(fshift))
                frame_file_name = 'frame{}_{}_{}_{}.jpg'.format(video_count, frame_count, j, label)
                df = df.append({'frame': frame_file_name, 'label': label}, ignore_index=True)
                df["label"] = df["label"].astype(int)
                # Create FFT representation of face
                cv2.imwrite('frames_fft/' + frame_file_name, magnitude_spectrum)
                # Only get the first face in the image
                break

        video_count = video_count + 1
        if os.path.isfile('labels.csv'):
            print("appending to csv")
            df.to_csv('labels.csv', header=None, mode='a')
        else:
            print("Creating new csv")
            df.to_csv('labels.csv')


def clear_fft_directory():
    for filename in os.listdir('frames_faces'):
        os.remove('frames_faces/' + filename)
    for filename in os.listdir('frames_fft'):
        os.remove('frames_fft/' + filename)


clear_fft_directory()
print("cleared")
# Generate FFT images from youtube data
#generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/manipulated_sequences/Deepfakes/c23/videos/', 0, 5)
#generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/original_sequences/youtube/c23/videos/', 1, 5)

# Generate FFT images from actor data
generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/manipulated_sequences/DeepFakeDetection/c23/videos/', 0, 5)
generateFFT('/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/original_sequences/actors/c23/videos/', 1, 5)



