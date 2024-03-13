import os
import json
import numpy as np
import cv2


# List of the cameras we're interested in.
camera_numbers = [00, 27, 29]
# Lambda function to format the image path
format_image = lambda  id ,cam: f"00_{cam}/00_{cam}_{id}.jpg"

def image_path(subj, id, cam):
    return f"panoptic/{subj}/hdImgs/00_{cam}/00_{cam}_{id}.jpg"

# List of the subjects we're interested in. They've been selected manually by looking at the catalog
subjects = ['161029_flute1',
            '161029_piano1',
            '170407_office2',
            '170915_office1',
            '171026_pose1',
            '171026_pose2',
            '171026_pose3',
            '171204_pose1',
            '171204_pose2',
            '171204_pose3',
            '171204_pose4',
            '171204_pose5',
            '171204_pose6']

# Opens a json file and returns the body element
def extract_body(file):
    with open(file, "r") as f:
        data = json.load(f)

    return data['bodies']

# Divides a list of joints into n-sized lists, since panoptic returns a one-dimensional list of 76 elements
def divide_joints(joints, n=4): 
    return [joints[i:i + n] for i in range(0, len(joints), n)]  

# Receives a dictionary like the ones returned by join_data and returns a list of dictionaries:
# Each dictionary contains the id, joints19, camera parameters and image path
def expand_data(elem, cameras):
    expanded = [] 
    for cam in cameras:
        expanded.append({
            'id' : elem['id'],
            'body_id' : elem['body_id'],
            'joints19' : elem['joints19'],
            'img' : os.path.join(elem['img_folder'], format_image(elem['id'], cam)),
            'calibration' : elem['calibration'],
            'camera' : cam
        })
          
# Retrieve the camera parameters from the calibration file, given a subject, a list of cameras and a root directory
def retrieve_cameras (subject, root_dir = "", camera_numbers = camera_numbers):
    cameras = {}
    for cam in camera_numbers:
        with open(os.path.join(root_dir, f"{subject}/calibration_{subject}.json"), "r") as f:
            data = json.load(f)
            retireved_cameras = data['cameras']
            interesting_cameras = [camera for camera in retireved_cameras if camera['type'] == "hd" and camera['node'] == cam]
            if len(interesting_cameras) != 1:
                raise ValueError(f"Error: {len(interesting_cameras)} cameras found for camera number {cam} in subject {subject}")
            else:
                cameras[cam]= {
                    'K' : interesting_cameras[0]['K'],
                    'distCoef' : interesting_cameras[0]['distCoef'],
                    'R' : interesting_cameras[0]['R'],
                    't' : interesting_cameras[0]['t']
                    }
    return cameras


# Return list of dictionaries by subject from a root folder that connects id, joints file, calibration file and image folder
def join_data(subjects = subjects, root_dir = "e:/tfg/panoptic"):
    files = []

    for subj in subjects:
            json_folder = f"{subj}/hdPose3d_stage1_coco19"
            number_of_json = lambda name: name[-13:-5]

            f = [f"{json_folder}/{f}" for f in os.listdir(os.path.join(root_dir,json_folder)) if f.endswith('.json')]

            for file in f:
                
                if body := extract_body(os.path.join(root_dir, file)):
                     body = body[0]
                else:
                    continue
                files.append({
                     'subj' : subj,
                     'id': number_of_json(file),
                     'body_id' : body['id'],
                     'joints19' : body['joints19'],
                     'img_folder' : f"{subj}/hdImgs",
                     'calibration' : f"{subj}/calibration_{subj}.json"
                })
            
    return files


# Projects 3D points into 2D points using the camera parameters K and distortion coefficient, and optionally the rotation and translation vectors
def project2d(points, K, distCoef, rvec = None, tvec = None):
    if rvec is None: # If no rotation vector is given, assume no rotation
        rvec = np.zeros((3, 1))
    if tvec is None: # If no translation vector is given, assume no translation
        tvec = np.zeros((3, 1))  

    points = np.array(points.copy()) 
    K = np.array(K.copy())
    distCoef = np.array(distCoef.copy())

    points_2d, _ = cv2.projectPoints(points, rvec, tvec, K, distCoef)

    # Reshape the points to the format we're using
    return points_2d.reshape(-1, 2)