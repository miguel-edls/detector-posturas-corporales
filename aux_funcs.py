import pickle
import numpy as np



#This two functions are used to save and load objects in the form of pickles
def to_pickle(obj, filename):
    with open('pickles/'+filename, 'wb') as handle:
        pickle.dump(obj, handle)

def from_pickle(filename):
    with open('pickles/'+filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def cam2pixel(image):
    param = image.copy()
    cam_coord = np.array(param['keypoints'])
    f = param['f']
    c = param['c']
    
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2] 
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def cam2pixel2(kpts, f, c):
    cam_coord = np.array(kpts)
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1] 
    img_coord = np.concatenate((x[:,None], y[:,None]),1)
    return img_coord

def world2cam2(image):
    kpts = np.array(image['npkeypoints'])
    R = np.array(image['R'])
    t = np.array(image['t'])
    cam_coord = np.dot(R, kpts.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return np.array([[x, -y, z] for [x, y, z] in cam_coord])



def world2cam(kpts, R, t):
    try:
        kpts = np.array([[x, y, z] for [x, y, z, _] in kpts])
    except:
        kpts = np.array(kpts)
    R = np.array(R)
    t = np.array(t)
    cam_coord = np.dot(R, kpts.T).T + t.reshape(1,3)
    return [[x, -y, z] for [x, y, z] in cam_coord]


