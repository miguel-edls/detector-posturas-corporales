import time
import numpy as np
import math as m
import mediapipe as mp
from mmpose.apis import MMPoseInferencer
from mmcv.image import imread
import matplotlib.pyplot as plt
from PIL import Image

#Método para calcular el ángulo entre un vector definido por dos puntos y el eje vertical
def angle_with_vertical(p1, p2):
    
    #Si los vectores no tienen tres componentes, lanzamos un error
    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("The vectors must have three components")
    
    # Define los dos puntos en el espacio 3D
    punto1 = np.array(p1)  
    punto2 = np.array(p2)  

    # Calcula la dirección de la recta
    direccion = punto2 - punto1
    direccion = direccion / np.linalg.norm(direccion)

       
    eje_y = np.array([0, 1, 0])

    # Calcula el angulo
    angulo_rad = np.arccos(np.clip(np.dot(direccion, eje_y), -1.0, 1.0))

    
    angulo_grados = np.degrees(angulo_rad)

    return angulo_grados

#Método para mostrar una imagen con las predicciones de mediapipe
def show_mediapipe(item):
    skeleton = (
    (0, 2), (2, 7), (0, 5), (5, 8),(9, 10), #Nose, eyes, ears and mouth
    (11, 12), (11, 23), (23, 24), (12, 24), #Thorax
    (23, 25), (25, 27), (27, 29), (29, 31), #Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), #Right leg
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), #Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22) #Right arm
    )

    plt.figure(figsize=(11, 11))
    
    img = np.flipud(Image.open(item['img']))
    
    
    plt.imshow(img, origin="lower")
    
    keypoints = np.array(item['blaze_kpts'])
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    plt.scatter(x, y, marker=".")
    for i, j in list(skeleton):
        plt.plot([x[i], x[j]], [y[i], y[j]], c='c')
    plt.show()
    

#Método para mostrar una imagen con las predicciones de mmpose
def show_mmpose(item, mx=1, my=1):
    skeleton = (
        (6, 5), (5, 4), (4, 0), #Right leg
        (3, 2), (2, 1), (1, 0), #Left leg
        (0, 7), (7, 8), (8, 9), (9, 10), #Spine and head
        (8, 11), (11, 12), (12, 13) , #Right arm
        (8, 14), (14, 15), (15, 16) #Left arm
    )

    plt.figure(figsize=(11, 11))
    
    img = np.flipud(Image.open(item['img']))
    
    
    plt.imshow(img, origin="lower")
    
    keypoints = np.array(item['mmpose_kpts'])
    x = [(1-p)*1000*mx+560 for p in keypoints[:, 0]]
    y = [p*1000*my for p in keypoints[:, 2]]
    plt.scatter(x, y, marker=".")
    for i, j in list(skeleton):
        plt.plot([x[i], x[j]], [y[i], y[j]], c='c')
    plt.show()

#Método para calcular el punto medio entre dos puntos
def get_midpoint(p1, p2):
    midpoint = [(p1.x + p2.x) / 2, 
                (p1.y + p2.y) / 2, 
                (p1.z + p2.z) / 2]
    
    return midpoint

#Método para calcular el vector entre un punto y el origen, y normalizar el eje z si es necesario
def get_vector(landmark1, z_normalization=1):
    vector = [landmark1.x, 
              landmark1.y, 
              landmark1.z * z_normalization]
    
    return vector

#Método para calcular el ángulo desde una predicción de mediapipe
def get_angle_mp(pose_landmarks):
    # Calculate the midpoint between landmarks 11 and 12
    midpoint = get_midpoint(pose_landmarks[11], pose_landmarks[12])

    # Calculate the vector from landmark 0 to the midpoint
    vector = get_vector(pose_landmarks[0], z_normalization=0.5)

    # Calculate the angle between the vector and the vertical axis
    angle = angle_with_vertical(vector, midpoint)

    return angle

#Método para calcular el ángulo desde una predicción de mmpose
def get_angle_mmpose(result):
    neck = np.array(result['predictions'][0][0]['keypoints'][8])
    nose = np.array(result['predictions'][0][0]['keypoints'][9])

    angle = angle_with_vertical(nose, neck)

    return angle

def get_mediapipe_keypoints(items):


    #iniciamos el temporizador
    start_time = time.time()

    #Cargamos el modelo de mediapipe y las opciones
    model_path = 'models/pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    #Abrimos el modelo de mediapipe y ejecutamos las predicciones
    with PoseLandmarker.create_from_options(options) as landmarker:

        for item in items:

            image = mp.Image.create_from_file(item['img'])
        
        
        

            results = landmarker.detect(image)
            item['blaze_results'] = results
        

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")


#Cargamos el modelo de mmpose
def get_mmpose_keypoints(items):
    inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")
    start_time = time.time()

    #Ejecutamos las predicciones con MMPose
    for item in items:
        
        result_generator = inferencer(item['img'], return_datasamples=True)
        result = next(result_generator)
        item['mmpose_results'] = result

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
