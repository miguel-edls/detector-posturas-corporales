import time
import numpy as np
import math as m
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def angle_with_vertical(p1, p2):
    
    #Si los vectores no tienen tres componentes, lanzamos un error
    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("The vectors must have three components")
    
    # Define los dos puntos en el espacio 3D
    punto1 = np.array(p1)  # Reemplaza x1, y1, z1 con las coordenadas del primer punto
    punto2 = np.array(p2)  # Reemplaza x2, y2, z2 con las coordenadas del segundo punto

    # Calcula la dirección de la recta
    direccion = punto2 - punto1

    # Proyecta la dirección en el plano XY (ignora la coordenada Z)
    proyeccion_xy = np.array([direccion[0], direccion[1], 0])

    # Calculamos el ángulo con el eje X en el plano XY
    eje_x = np.array([1, 0, 0])

    # Calcula el ángulo
    angulo_rad = np.arccos(np.dot(proyeccion_xy, eje_x) / (np.linalg.norm(proyeccion_xy) * np.linalg.norm(eje_x)))
    angulo_grados = np.degrees(angulo_rad)

    return angulo_grados





def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks = detection_result.pose_landmarks.landmark
    annotated_image = np.copy(rgb_image)

    # Print the 3D values of all the landmarks
        # Calculate the midpoint between landmarks 11 and 12
    midpoint = [(pose_landmarks[11].x + pose_landmarks[12].x) / 2, 
                (pose_landmarks[11].y + pose_landmarks[12].y) / 2, 
                (pose_landmarks[11].z + pose_landmarks[12].z) / 2]

    # Calculate the vector from landmark 0 to the midpoint
    vector = [midpoint[0] - pose_landmarks[0].x, 
              midpoint[1] - pose_landmarks[0].y, 
              midpoint[2] - pose_landmarks[0].z]

    # Calculate the angle between the vector and the vertical axis
    angle = m.degrees(m.acos(vector[1] / m.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)))

    # Print the angle
    # Draw the pose landmarks.
    height, width, _ = annotated_image.shape
    start_point = (int(pose_landmarks[0].x * width), int(pose_landmarks[0].y * height))
    end_point = (int(midpoint[0] * width), int(midpoint[1] * height))
    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks[:13]
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 4), (4, 6), (6, 8), (9, 10), (11, 12)],
        solutions.drawing_styles.get_default_pose_landmarks_style())
    # Create a new image for the side projection
    side_projection = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the landmarks on the side projection
    # Find the minimum and maximum z-coordinate
    

    # Draw the landmarks on the side projection
    for i, landmark in enumerate(pose_landmarks[:13]):
      # Normalize the z-coordinate to the range [0, 1]
      z = -int(landmark.z * width)
      y = int(landmark.y * height)
      cv2.circle(side_projection, (z, y), 2, (0, 255, 0), -1)
      cv2.putText(side_projection, str(i), (z + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    # Concatenate the annotated image and the side projection horizontally
    combined_image = np.concatenate((annotated_image, side_projection), axis=1)


    CoordinateListSingleton.getInstance().add_coordinates(midpoint[0], midpoint[1], midpoint[2], angle)
    avg_x, avg_y, avg_z, avg_angle = CoordinateListSingleton.getInstance().get_average()

    print(f"X: {avg_x:.3f} Y: {avg_y:.3f} Z: {avg_z:.3f} Angle: {avg_angle:.3f}", end='\r', flush=True)

    return combined_image

"""
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image
"""
model_path = 'models/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)



# Open the default camera
cap = cv2.VideoCapture(0)
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

x_values = []
y_values = []
z_values = []
angle_values = []


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw the pose annotation on the image.
    image = draw_landmarks_on_image(frame, results)

    # Display the resulting frame
    cv2.imshow('MediaPipe Pose', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# Check if the camera is opened successfully
if not cap.isOpened():
    print("Unable to open the camera")
    exit()

# Read the frame from the camera
ret, frame = cap.read()
# cv2.imshow("Webcam Image", frame)
# cv2.waitKey(0)
cv2.imwrite('foto2.jpg', frame)

# image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

image = mp.Image.create_from_file('foto2.jpg')

# Load the MediaPipe Pose model
detector = vision.PoseLandmarker.create_from_options(options)

detection_result = detector.detect(image)

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()