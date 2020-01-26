from mtcnn_pytorch.src import detect_faces, show_bboxes
from PIL import Image
import cv2

"""
This code uses the following repo, implementing MTCNN FACIAL DETECTION: https://github.com/TropComplique/mtcnn-pytorch
"""

def MTCNNFacialDetection(image, video_number, frame_number, label):
    image = image.convert('RGB')
    bounding_boxes, landmarks = detect_faces(image)
    face_number = 0
    cropped_faces = []
    for b in bounding_boxes:
        cropped_face = image.crop((b[0], b[1], b[2], b[3]))
        cropped_face.convert('LA')
        cropped_faces.append(cropped_face)
        face_number += 1
    return cropped_faces

    #image_boxes = show_bboxes(image, bounding_boxes, landmarks)


