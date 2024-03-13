import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from transforms import rotate_to_profile
from calculations import get_angle_2d



def detect_posture_from_file(imagepath = r"webcam_imgs\foto_encorvado.jpg"):
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(imagepath)

    detection_result = detector.detect(image)
    width = image.width
    height = image.height

    
    try:
        points = [[point.x*width, (1-point.y)*height+30, point.z*width] for point in detection_result.pose_landmarks[0]]
    except:
        return []


    

    return points

def detect_posture(image, detector):

    rgb_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)

    detection_result = detector.detect(rgb_frame)
    width = rgb_frame.width
    height = rgb_frame.height

    
    try:
        points = [[point.x*width, (1-point.y)*height+30, point.z*height] for point in detection_result.pose_landmarks[0]]
    except:
        return []

    return points


def analyze_posture(keypoints):
    keypoints = keypoints.copy()
    rotated = rotate_to_profile(keypoints, [keypoints[11], keypoints[12]])
    se_angle = get_cva(rotated)
    # Usamos el coeficiente y la intersección de la regresión lineal para obtener el CVA en lugar del modelo entero
    coef_ = -0.98474693
    intercept_ = 64.17786676378498
    # Retornamos el complemento a 90 del CVA calculado (el CVA real)
    return (coef_*se_angle + intercept_)



def print_posture(imagepath = r"webcam_imgs\foto_encorvado.jpg"):
    img = cv2.imread(imagepath)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    plx, ply, plz = detect_posture(imagepath)

    ax.imshow(img[:, :, ::-1])
    ax.plot(plx, ply, 'r,')
    plt.show()
    

def get_cva(keypoints):
    keypoints = keypoints.copy()
    rsmp = (np.array(keypoints[12]) + np.array(keypoints[11]))/2
    remp = (np.array(keypoints[7]) + np.array(keypoints[8]))/2
    se_angle = get_angle_2d(remp, rsmp)
    return se_angle