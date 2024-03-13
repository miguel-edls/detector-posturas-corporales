import cv2
from mmpose.apis import MMPoseInferencer, inference_topdown
import numpy as np

img_path = 'foto2.jpg'   

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose3d = 'human3d')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer('webcam', show=False)
result = next(result_generator)
#results = [result for result in result_generator]

#        0: 'left_shoulder',
#        1: 'right_shoulder',
#        2: 'left_elbow',
#        3: 'right_elbow',
#        4: 'left_wrist',
#        5: 'right_wrist',
#        6: 'left_hip',
#        7: 'right_hip',
#        8: 'left_knee',
#        9: 'right_knee',
#        10: 'left_ankle',
#        11: 'right_ankle',
#        12: 'top_head',
#        13: 'neck'


while(True):
    neck = np.array(result['predictions'][0][0]['keypoints'][1])
    nose = np.array(result['predictions'][0][0]['keypoints'][0])

    # Calculate the vector from the neck to the nose
    vector_neck_to_nose = neck - nose

    # Define the vertical vector
    vertical_vector = np.array([0, 0, 1])

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_neck_to_nose, vertical_vector)

    # Calculate the magnitudes of the vectors
    magnitude_neck_to_nose = np.linalg.norm(vector_neck_to_nose)
    magnitude_vertical = np.linalg.norm(vertical_vector)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_neck_to_nose * magnitude_vertical)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    print("Angle: ", angle_degrees)

    result = next(result_generator)


print('hola')