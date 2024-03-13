import numpy as np
from scipy.spatial.transform import Rotation as R
from calculations import get_angle_with_YZ, get_angle_2d
import cv2


def to_180(angle):
    if angle>180:
        return angle-360
    else:
        return angle

def world_to_cam(kpts, R, t):
    # We add the .squeeze() method to the dot product to remove the extra dimension in the resulting numpy array
    cam_coord = np.dot(R, kpts.transpose(1,0)).squeeze().transpose(1,0) + t.reshape(1,3)

    return np.array([[x, -y, z] for [x, y, z] in cam_coord])


def to_perspective(image_path = 'webcam_imgs/foto_encorvado.jpg', perspective = [[100, 50], [400, 50], [60, 300], [450, 300]]):
    
    image = cv2.imread(image_path)
    
    src_points = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
    dst_points = np.float32(perspective)
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective_transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    
    cv2.imwrite('saved.png', perspective_transformed_image)

def get_rotated_cva(image):
    image = image.copy()
    c7 = image['c7']
    tr = image['tr']
    dir_vector = [image.keypoints[14], image.keypoints[11]]

    rc7, rtr = rotate_to_profile([c7, tr], dir_kpts=dir_vector)
    cva = get_angle_2d(rc7, rtr)

    return cva

def get_cva2(image):
    image = image.copy()
    c7 = image['c7']
    tr = image['tr']

    cva = get_angle_2d(c7, tr)

    return cva

def extract_angles(img):
    img = img.copy()
    kpts = img['keypoints']

    smp = (np.array(kpts[14]) + np.array(kpts[11]))/2
    smp_n = get_angle_2d(smp, kpts[8])
    n_ch = get_angle_2d(kpts[8], kpts[9])
    ch_th = get_angle_2d(kpts[9], kpts[10])
    smpth = get_angle_2d(smp, kpts[10])
    return{"smpn" : smp_n, 'nch' : n_ch, 'chth': ch_th, 'smpth': smpth}

def get_cvv(image):
    image = image.copy()
    c7 = image['c7']
    tr = image['tr']
    dir_vector = [image.keypoints[14], image.keypoints[11]]

    rc7, rtr = rotate_to_profile([c7, tr], dir_kpts=dir_vector)
    cva = rtr - rc7

    return cva


def extract_rotated_angles(img):
    img = img.copy()

    kpts = rotate_to_profile(img['keypoints'])

    smp = (kpts[14] + kpts[11])/2

    smp_n = get_angle_2d(smp, kpts[8])
    n_ch = get_angle_2d(kpts[8], kpts[9])
    ch_th = get_angle_2d(kpts[9], kpts[10])
    smpth = get_angle_2d(smp, kpts[10])
    return{"smpn" : smp_n, 'nch' : n_ch, 'chth': ch_th, 'smpth': smpth}
    

def extract_rotated_vectors(img):
    img = img.copy()

    kpts = rotate_to_profile(img['keypoints'])

    smp = (kpts[14] + kpts[11])/2

    smp_n = kpts[8] - smp
    n_ch = kpts[9] - kpts[8]
    ch_th = kpts[10] - kpts[9]
    return{"smpnx" : smp_n[0], 'smpny' : smp_n[1],  'nchx' : n_ch[0], 'nchy' : n_ch[1], 'chthx': ch_th[0], 'chthy': ch_th[1]}


def get_tagged_cva(image):
    c7 = image['c7'].copy()
    c7.append(0)
    tr = image['tr'].copy()
    tr.append(0)

    all_kpts = image['keypoints'].copy()
    all_kpts.append(c7)
    all_kpts.append(tr)

    kpts_c7_tr = rotate_to_profile(all_kpts)
    c7, tr = kpts_c7_tr[-2:]
    cva = get_angle_2d(c7, tr)

    return cva
    

def rotate_to_profile(keypoints, dir_kpts = None):
    keypoints = keypoints.copy()
    if dir_kpts is None:
        angle = get_angle_with_YZ(keypoints[14], keypoints[11])
    else:
        angle = get_angle_with_YZ(dir_kpts[0], dir_kpts[1])
    
    rotated_keypoints = rotate_Y(keypoints, -angle)
    return rotated_keypoints

def rotate_Y(pts, angle):
    points = [elem + [0] if len(elem) == 2 else elem for elem in pts.copy()]
    rotation_vector = np.array([0, 1, 0]) * angle
    # Create a rotation object from the rotation vector
    rotation = R.from_rotvec(rotation_vector)
    # Apply rotation to all points
    rotated_points = rotation.apply(points)
    return rotated_points

def get_projected_angle(v, n):
    n = normalize(n)
    proj_n_v = (np.dot(v, n) / n**2) * n
    v_plane = v - proj_n_v
    return v_plane

def normalize(vector):
    vector = np.array(vector)
    return vector / np.linalg.norm(vector)