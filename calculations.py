import numpy as np


def get_angle_2d(p1, p2, threesixty = False):
    p1 = p1[:2]
    p2 = p2[:2]
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = np.arctan2(delta_x, delta_y)
    
    #to degrees
    angle = np.degrees(angle)
    if threesixty:
        if angle < 0:
            angle = 360 + angle
    return angle
      

def get_shoulder_angles(keypoints,  shoulder_indexes = [14, 11]):

    shoulder_vector = np.array(keypoints[shoulder_indexes[0]]) - np.array(keypoints[shoulder_indexes[1]])
    return np.arctan2(shoulder_vector[0], shoulder_vector[2])


def get_angle_with_YZ(point1, point2):
    dx = point2[0] - point1[0]
    dz = point2[2] - point1[2]
    angle = np.arctan2(dx, dz)
    return angle

def distance2d(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def distance3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
def iq_range(arr):
    q75, q25 = np.percentile(arr, [75 ,25])
    
    return q75 - q25

def trimmed_mean(arr, trim = 0.1):
    arr = np.sort(arr)
    trim = int(len(arr)*trim)
    return np.mean(arr[trim:-trim])