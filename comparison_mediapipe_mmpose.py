import time
import numpy as np
import math as m
import mediapipe as mp
from mmpose.apis import MMPoseInferencer

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

    image_recto = mp.Image.create_from_file('samples/foto_recto.jpg')
    image_medio = mp.Image.create_from_file('samples/foto_medio.jpg')
    image_encorvado = mp.Image.create_from_file('samples/foto_encorvado.jpg')

    
    start_time = time.time()

    results1 = landmarker.detect(image_recto)
    print("Angulo 1:")
    print(get_angle_mp(results1.pose_world_landmarks[0]))
    results2 = landmarker.detect(image_medio)
    print("Angulo 2:")
    print(get_angle_mp(results2.pose_world_landmarks[0]))
    results3 = landmarker.detect(image_encorvado)
    print("Angulo 3:")
    print(get_angle_mp(results3.pose_world_landmarks[0]))

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")


#Cargamos el modelo de mmpose
inferencer = MMPoseInferencer(pose3d = 'human3d')

#Ejecutamos las predicciones con MMPose
start_time = time.time()
result_generator = inferencer('samples/foto_recto.jpg', show=False)
result = next(result_generator)
print("Angulo 1:")
print(get_angle_mmpose(result))
result_generator = inferencer('samples/foto_medio.jpg', show=False)
result = next(result_generator)
print("Angulo 2:")
print(get_angle_mmpose(result))
result_generator = inferencer('samples/foto_encorvado.jpg', show=False)
result = next(result_generator)
print("Angulo 3:")
print(get_angle_mmpose(result))

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")