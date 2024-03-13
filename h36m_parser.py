import numpy as np
import json
from aux_funcs import cam2pixel, world2cam2, cam2pixel2, world2cam






# Retrieves the 3d points of an image object and returns the pixel space projected 2d points
def get_camera_points_3d(image):
    camera = image['camera']
    f = camera['f']
    c = camera['c']
    res = cam2pixel(world2cam2(image), f, c)

    res = [[int(num) for num in point] for point in res]
    
    return res

# Retrieves
def get_parameters(image):

    intrinsics, extrinsics = load_parameters()

    subject = f"S{image['subject']}"

    extrinsics = extrinsics[subject]

    R = di(image['cam_idx'], extrinsics)['R']
    t = di(image['cam_idx'], extrinsics)['t']

    calibration = di(image['cam_idx'], intrinsics)['calibration_matrix']

    return R, t, calibration


# Load both intrinsic and extrinsic camera parameters from json file
def load_parameters():
    params = None
    with open('camera-parameters.json') as file:
        params = json.load(file)  
    intrinsics = params['intrinsics']
    extrinsics = params['extrinsics']
    return intrinsics, extrinsics


# Returns dictionary value from positional index. ATENCIÓN: NO FUNCIONARÁ EN VERSIONES DE PYTHON ANTERIORES A 3.7
def di(n, dictionary):
    return dictionary[list(dictionary.keys())[int(n)-1]]

