import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from aux_funcs import to_pickle, from_pickle, cam2pixel, world2cam, cam2pixel2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy.spatial.transform import Rotation as R
from transforms import rotate_to_profile, normalize
from panoptic_parser import project2d

skeleton_h36m = (
        (6, 5), (5, 4), (4, 0), #Right leg
        (3, 2), (2, 1), (1, 0), #Left leg
        (0, 7), (7, 8), (8, 9), (9, 10), #Spine and head
        (8, 11), (11, 12), (12, 13) , #Right arm
        (8, 14), (14, 15), (15, 16) #Left arm
    )

skeleton_panoptic = (
    (8, 7), (7, 6), (6, 2), 
    (14, 13), (13, 12), (12, 2),
    (2, 0), (0, 1),
    (0, 3), (3, 4), (4, 5),
    (0, 9), (9, 10), (10, 11),
    (1, 15), (15, 16),
    (1, 17), (17, 18)
    
)

skeleton_blaze = (
    (0, 2), (2, 7), (0, 5), (5, 8),(9, 10), #Nose, eyes, ears and mouth
    (11, 12), (11, 23), (23, 24), (12, 24), #Thorax
    (23, 25), (25, 27), (27, 29), (29, 31), #Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), #Right leg
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), #Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22) #Right arm
)

def get_picture(image, alpha):
    img = Image.open("images/"+image['file_name'])

    arrayimage = np.flipud(np.array(img) /255)
    arrayimage = np.dstack((arrayimage, np.zeros((arrayimage.shape[0], arrayimage.shape[1]))+alpha))

    width = arrayimage.shape[0]
    height = arrayimage.shape[1]
    # Paso 2: Creamos una malla de puntos en el plano XY
    x = np.linspace(0, height-1, height)
    y = np.linspace(0, width-1, width)
    x, y = np.meshgrid(x, y)

    # Paso 3: Creamos una matriz de ceros para el eje Z
    z = np.zeros(x.shape)

    return arrayimage, x, y, z, width, height



def visualize_image(image, defin = 50):

    arrayimage, x, y, z, width, height = get_picture(image)

    # Paso 4: Creamos una figura y un eje 3D
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Paso 5: Creamos la superficie con los datos de la malla y la imagen

    ax.plot_surface(x, y, z, rcount = defin, ccount = defin, facecolors = arrayimage) 

    ax.view_init(elev=60, azim=130, roll=-140)
    
    ax.set_xlim([0, height])
    ax.set_ylim([0, width])
    #plt.axis('equal')

    # Disable autoscale
    ax.autoscale(enable=False)
    # Optional: labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Simple Plane')

    

    plt.show()


def show(image):
    try:
        img = Image.open(image['file_name'])
    except:
        img = Image.open(image['img'])
    plt.imshow(img)
    plt.show()




def show2d(img):

    fig = plt.figure()
    fig.set_size_inches(9, 9)

    image = Image.open(img['file_name'])
    plt.imshow(image)

    img_coord = cam2pixel(np.array(img['keypoints'] ), img['camera']['f'], img['camera']['c'])
    plt.scatter(img_coord[:,0], img_coord[:,1], marker=".")

    
    plt.show()



def midpoint(p1, p2):
    return (np.array(p2) + np.array(p1))/2


def show2d_pan(image, view = "side", vtype="truth"):
    
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    img = Image.open(file)
    plt.imshow(img)
    img_coord = project2d(np.array(kpts), cam['K'], cam['distCoef'])
    plt.scatter(img_coord[:,0], img_coord[:,1], marker=".")
    plt.show()

def visualize_flat(image, keypoints = None, skeleton = skeleton_panoptic, show_cva = False, view = "side", vtype="truth"):

    image = image.copy()

    if view == "side":
        file = f"panoptic/{image['subj']}/hdImgs/00_27/00_27_{image['framenum']}.jpg"
        cam = image['cams'][27]
    elif view == "front":
        show_cva = False
        file = f"panoptic/{image['subj']}/hdImgs/00_00/00_00_{image['framenum']}.jpg"
        cam = image['cams'][0]
    else:
        raise ValueError("Invalid view")
    if vtype == "truth":
        kpts = world2cam(image['keypoints'], cam['R'], cam['t'])
        img_coord = np.array(project2d(kpts, cam['K'], cam['distCoef']))
    elif vtype == "predicted":
        skeleton = skeleton_blaze
        if view == "side":
            img_coord = np.array(image['keypoints_predicted_side'])
        else:
            img_coord = np.array(image['keypoints_predicted_front'])
    else:
        raise ValueError("Invalid vtype")

    plt.figure(figsize=(11, 11))
    
    img = np.flipud(Image.open(file))
    #img = Image.open(file)
    
    plt.imshow(img, origin="lower")


    
    
    x = img_coord[:, 0]
    y = img_coord[:, 1]
    y = [elem-30 for elem in y]

    if keypoints is not None:
        x_reduced = []
        y_reduced = []
        for i in keypoints:
            x_reduced.append(x[i])
            y_reduced.append(y[i])
        x = x_reduced
        y = y_reduced
        skeleton = [(keypoints.index(i), keypoints.index(j)) for i, j in skeleton if i in keypoints and j in keypoints]

    plt.scatter(x, y, marker=".")

    if show_cva:
        c7 = image['c7']
        tr = image['tr']
        cva_x = (c7[0], tr[0])
        cva_y = (c7[1],tr[1])
        plt.plot(cva_x, cva_y, 'r')

    for i, j in list(skeleton):
        plt.plot([x[i], x[j]], [y[i], y[j]], c='c')
    plt.show()

def plot_keypoints(keypoints, skeleton = skeleton_panoptic, show_skeleton = True, cva = None, title = "Proyección frontal", y= 1):
    
    try:
        x = keypoints[:, 0]
        y = keypoints[:, y]
    except:
        x = [kp[0] for kp in keypoints]
        y = [kp[y] for kp in keypoints]

    plt.gca().set_aspect('equal')


    plt.scatter(x, y, c='blue', label='Keypoints')

    if show_skeleton:
        for i, j in list(skeleton):
            plt.plot([x[i], x[j]], [y[i], y[j]], c='c')

    if cva is not None:
        c7, tr = cva
        plt.plot([c7[0], tr[0]], [c7[1], tr[1]], c='r')

    plt.title(title)
    plt.show()

def get_neck_projection(image):
    img_coord = cam2pixel(np.array(image['keypoints']), image['camera']['f'], image['camera']['c'])
    shoulder_midpoint = (img_coord[11] + img_coord[14])/2
    sm_x = shoulder_midpoint[0]
    sm_y = shoulder_midpoint[1]
    x_coords = img_coord[:,0]
    y_coords = img_coord[:,1]

    return (sm_x, x_coords[10]), (sm_y, y_coords[10])
    

def compare_angles(image, show_cva = False):

    plt.figure(figsize=(9, 9))

    img = np.flipud(Image.open("images/"+image['file_name']))
    x_coords, y_coords = get_neck_projection(image)

    tragus = image['tr']
    c7 = image['c7']

    cva_x = (tragus[0], c7[0])
    cva_y = (tragus[1], c7[1])

    plt.imshow(img, origin='lower')
    plt.plot(x_coords, y_coords, 'r')
    plt.plot(cva_x, cva_y, 'b')

    plt.show()
    
    if show_cva:
        cva_calc = get_angle_2d(c7, tragus)
        cva_stored = image['cva']
        print(f"Stored CVA: {cva_stored} Calculated CVA: {cva_calc}")








def plot_rotated(image, skeleton = skeleton_h36m,  cva = False):
    image = image.copy()
    keypoints = image['keypoints']
    transform_points = [keypoints[14], keypoints[11]]
    keypoints = rotate_to_profile(keypoints)
    

    
    

    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]




    

    
    plt.scatter(x_coords, y_coords, c='blue', label='Keypoints')

    for i, j in list(skeleton):
        plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], c='c')


    if cva:

        c7, tr = rotate_to_profile([image['c7'], image['tr']], dir_kpts=transform_points)
        dir_vector = normalize(np.array(tr) - np.array(c7))[:2]
        origin = keypoints[8][:2] - np.array([100, 0])
        end = origin + dir_vector*150
        print("CVA: ", image['cva'])
        print('SMPTN: ', image['smpn'])
        print('NCH: ', image['nch'])
        print('CHTH: ', image['chth'])
        print('SMPTH: ', image['smpth'])


        plt.plot([origin[0], end[0]], [origin[1], end[1]], c='r')

    plt.gca().set_aspect('equal')

    plt.title('Proyección de perfil (dcha)')
    plt.show()

def plot_distribution(df, column):
        df[column].hist()
        plt.title("Distribución de " + column)
        plt.xlabel(column)
        plt.ylabel("Frecuencia")
        plt.show()

def shoulder_angles(img):
    A = img['keypoints'][11]
    B = img['keypoints'][14]

    delta_x = B[0] - A[0]
    delta_z = B[2] - A[2]

    angle = np.arctan(delta_x/delta_z)
    #to degrees
    angle = angle*180/np.pi
    return angle
    



finals =  {1: (0, 1), 5: (0,3), 6: (0, 1), 7: (0,1), 8: (0,1), 9:(2,3)}
