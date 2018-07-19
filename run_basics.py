import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time

from api import PRN
from utils.write import write_obj_with_colors

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = False) 


# ------------- load data
image_folder = 'TestImages/AFLW2000/'
save_folder = 'TestImages/AFLW2000_results'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    # read image
    image = imread(image_path)

    # the core: regress position map    
    if 'AFLW2000' in image_path:
        mat_path = image_path.replace('jpg', 'mat')
        info = sio.loadmat(mat_path)
        kpt = info['pt3d_68']
        pos = prn.process(image, kpt) # kpt information is only used for detecting face and cropping image
    else:
        pos = prn.process(image) # use dlib to detect face

    # -- Basic Applications
    # get landmarks
    kpt = prn.get_landmarks(pos)
    # 3D vertices
    vertices = prn.get_vertices(pos)
    # corresponding colors
    colors = prn.get_colors(image, vertices)

    # -- save
    name = image_path.strip().split('/')[-1][:-4]
    np.savetxt(os.path.join(save_folder, name + '.txt'), kpt) 
    write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

    sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})
