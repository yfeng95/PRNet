import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time

from api import PRN
import argparse


def main(args):
    if args.isShow:
        args.isOpencv = True
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
    if args.isObj:
        from utils.write import write_obj
    if args.isPose:
        from utils.estimate_pose import estimate_pose

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU

    prn = PRN(is_dlib = args.isDlib, is_opencv = args.isOpencv) 


    # ------------- load data
    image_folder = args.inputFolder
    save_folder = args.outputFolder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):
        
        name = image_path.strip().split('/')[-1][:-4]
        
        # read image
        image = imread(image_path)

        # the core: regress position map    
        pos = prn.process(image) # use dlib to detect face


        if args.isObj or args.isShow:        
            # 3D vertices
            vertices = prn.get_vertices(pos)
            # corresponding colors
            colors = prn.get_colors(image, vertices)
            write_obj(os.path.join(save_folder, name + '.obj'), vertices, colors, prn.triangles) #save 3d face(can open with meshlab)

        if args.isKpt or args.isShow:
            # get landmarks
            kpt = prn.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt) 
        
        if args.isPose or args.isShow:
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose) 

        if args.isShow:
            # ---------- Plot
            image_pose = plot_pose_box(image, camera_matrix, kpt)
            cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            cv2.imshow('dense alignment', plot_vertices(image, vertices))
            cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputFolder', default='TestImages/', type=str,
                        help='input images folder path')
    parser.add_argument('-o', '--outputFolder', default='TestImages/results', type=str, 
                        help='output obj file')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=bool, 
                        help='whether to use dlib to detect face')
    parser.add_argument('--isOpencv', default=True, type=bool, 
                        help='whether to use opencv')
    parser.add_argument('--isObj', default=True, type=bool, 
                        help='whether to output .obj file')
    parser.add_argument('--isKpt', default=False, type=bool, 
                        help='whether to output key points')
    parser.add_argument('--isPose', default=False, type=bool, 
                        help='whether to output estimated pose')
    parser.add_argument('--isShow', default=False, type=bool, 
                        help='whether to show the results with opencv(need set isOpencv=True')
    main(parser.parse_args())