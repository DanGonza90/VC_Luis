import argparse
import numpy as np
import h5py
import scipy.io as sio
import sys
import cv2
from open3d import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# "python3  rigidtransforms.py  rgbimgs depthimgs cameracalib transforms"

description_text = "Given a sequence of images from a depth camera, the code computes the rigid" \
                   " transformations (position and orientation) of the camera to a given " \
                   "world coordinate system. The world coordinate frame is set to one of " \
                   "the camera frames. " \
 \
# Initiate the parser
parser = argparse.ArgumentParser(description=description_text)
parser.add_argument("rgbimgs", help="A string with the path to a file (text file) where its contents have N "
                                    "lines. Each line is a string with the path to each rgb image")
parser.add_argument("depthimgs", help="A string with the path to a file (text file) where its contents have "
                                      "N lines. Each line is a string with the path to each depth images")
parser.add_argument("cameracalib", help="The path to a file with the camera calibration parameters. The default file"
                                        " type is a .txt file. Use the optional argument -cf if you want to use a"
                                        " Matlab .mat file.")
parser.add_argument("transforms", help="String with the path to a file (text/ascii) where each row should be "
                                       "12 doubles with the transformation from image i to the world :"
                                       " Ri_11 Ri_12 ... Ri_33 Ti_x Ti_y Ti_z")
parser.add_argument("-cf", "--calibfile",
                    choices=["mat", "txt"],
                    default="txt",
                    help="The camera calibration file can either be a .mat or a .txt file.    "
                         "For a TXT file: The first row has 9 doubles with Krgb:"
                         " k11 k12 k13 k21 k22 k23 k31 k32 k33; "
                         "Second row has 9 double with Kdepth:"
                         " k11 k12 k13 k21 k22 k23 k31 k32 k33; "
                         "Third row has 9 doubles with Rotation matrix: "
                         "R11 R12 .... R33; "
                         "Fourth row has 3 doubles with Translation vector: Tx Ty Tz. "
                         "For a MAT file: A structure with the instrinsic and extrinsic camera parameters. "
                         "cam_params.Kdepth  - the 3x3 matrix for the intrinsic parameters for depth "
                         "cam_params.Krgb - the 3x3 matrix for the intrinsic parameters for rgb "
                         "cam_params.R - the Rotation matrix from depth to RGB (extrinsic params) "
                         "cam_params.T - The translation from depth to RGB")

args = parser.parse_args()


class CamParamsMat:
    def __init__(self, mat):
        self.mat = mat
        self.Kdepth = np.zeros((3, 3))
        self.Krgb = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))

        self.get_krgb()
        self.get_kdepth()
        self.get_r()
        self.get_t()

    def get_krgb(self):
        self.Krgb = self.mat["RGB_cam"]['K'][0][0]

    def get_kdepth(self):
        self.Kdepth = self.mat["Depth_cam"]['K'][0][0]

    def get_r(self):
        self.R = self.mat["R_d_to_rgb"]

    def get_t(self):
        self.T = self.mat["T_d_to_rgb"]


class CamParamsTxt:
    def __init__(self, txt):
        self.txt = txt
        self.Kdepth = np.zeros((3, 3))
        self.Krgb = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))

        self.get_krgb()
        self.get_kdepth()
        self.get_r()
        self.get_t()

    def get_krgb(self):
        # First line of the text file
        aux = self.txt[0].split()
        k_values = np.array(list(map(float, aux)))
        self.Krgb = k_values.reshape((3, 3))

    def get_kdepth(self):
        # Second line of the text file
        aux = self.txt[1].split()
        k_values = np.array(list(map(float, aux)))
        self.Kdepth = k_values.reshape((3, 3))

    def get_r(self):
        # Third line of the text file
        aux = self.txt[2].split()
        r_values = np.array(list(map(float, aux)))
        self.R = r_values.reshape((3, 3))

    def get_t(self):
        # Forth line of the text file
        aux = self.txt[3].split()
        t_values = np.array(list(map(float, aux)))
        self.T = t_values.reshape((3, 1))


def rigid_transforms():
    Ri_11 = 11.321
    Ri_12 = 12.321
    Ri_13 = 13.321
    Ri_21 = 21.321
    Ri_22 = 22.321
    Ri_23 = 23.321
    Ri_31 = 31.321
    Ri_32 = 32.321
    Ri_33 = 33.321
    Ti_x = 40.321
    Ti_y = 41.321
    Ti_z = 42.321

    data = "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(Ri_11, Ri_12,
                                                                                                          Ri_13, Ri_21,
                                                                                                          Ri_22, Ri_23,
                                                                                                          Ri_31, Ri_32,
                                                                                                          Ri_33, Ti_x,
                                                                                                          Ti_y, Ti_z)
    with open(args.transforms, "a") as f:
        f.write(data)


def img_generator(paths_file):
    for path in paths_file:
        if path.endswith("\n"):
            path = path[:-1]
        img_ = cv2.imread(path)
        #img_ = open3d.io.read_image(path)
        yield img_


def depth_array_generator(paths_file):
    for path in paths_file:
        if path.endswith("\n"):
            path = path[:-1]
        depth_array_ = sio.loadmat(path)
        depth_array_ = depth_array_['depth_array']

        yield depth_array_


def lixo():
    pass
    #x = points3d.reshape((480 * 640, 3))
    #ev = o3d.visualization.Visualizer()
    #ev.create_window()
    #pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
    #ev.add_geometry(pcd)
    #ev.run()
    #ev.destroy_window()


def get_3dPoints(depth_array_, cam_params_):
    xyz1 = np.zeros((4, 1))
    total_xyz = np.zeros((depth_array_.shape[0]*depth_array_.shape[1],3))
    intrinsic = np.zeros((4, 4))
    extrinsic = np.zeros((4, 4))
    uv = np.ones((4, 1))
    t=0
    intrinsic[0:3,0:3] = cam_params_.Kdepth
    intrinsic[3,3] = 1

    extrinsic[0:3,0:3] = cam_params_.R
    extrinsic[0:3,3] = cam_params_.T.transpose()
    extrinsic[3, 3] = 1
    index = 0
    for a in range(depth_array_.shape[0]):
        for b in range(depth_array_.shape[1]):
            z = depth_array_[a, b]
            
            uv[0] = a
            uv[1] = b
            uv[2] = 1
            uv[3] = 1/z
            if z != 0:
                xyz1 = z * np.linalg.inv(intrinsic.dot(extrinsic)).dot(uv)
            else:
                xyz1 = np.zeros((4, 1))
            total_xyz[index] = [xyz1[0],xyz1[1],xyz1[2]]
            index += 1
    te = 0
    return total_xyz

'''
Import camera calibration files. Assign data to the cam_params structure (class)
'''
if args.calibfile == "mat" and args.cameracalib[-3:] == 'mat':
    try:
        mat = sio.loadmat(args.cameracalib)
    except FileNotFoundError:
        sys.exit("[Error]: Could not find the camera calibration file.")
    else:
        try:
            cam_params = CamParamsMat(mat)
        except ValueError:
            sys.exit('[Error]: Camera calibration file format invalid. Parse -h for help.')
        else:
            print("[INFO]: Camera calibration file loaded successfully")
elif args.calibfile == "txt" and args.cameracalib[-3:] == 'txt':
    try:
        with open(args.cameracalib, 'r') as p:
            calib_text = p.readlines()
    except FileNotFoundError:
        sys.exit("[Error]: Could not find the camera calibration file.")
    else:
        try:
            cam_params = CamParamsTxt(calib_text)
        except ValueError:
            sys.exit('[Error]: Camera calibration file format invalid. Parse -h for help.')
        else:
            print("[INFO]: Camera calibration file loaded successfully")
else:
    sys.exit('[Error]: Incompatible file extension parsed for the camera calibration file. Make sure the -cf argument '
             'and the file extension are the same. Parse -h for help.')

'''
Import rgb image files. Create a generator to yield one by one
'''
try:
    with open(args.rgbimgs, 'r') as p:
        rgb_paths = p.readlines()
except FileNotFoundError:
    sys.exit("[Error]: Could not find the rgb image paths file.")
else:
    rgb_gen = img_generator(rgb_paths)
    print("RGB image paths file loaded successfully")

'''
Import depth image files. Create a generator to yield one by one
'''
try:
    with open(args.depthimgs, 'r') as p:
        depth_paths = p.readlines()
except FileNotFoundError:
    sys.exit("[Error]: Could not find the depth image paths file.")
else:
    depth_gen = depth_array_generator(depth_paths)
    print("Depth image paths file loaded successfully")

while True:
    try:
        rgb = next(rgb_gen)
    except StopIteration:
        break
    else:
        cv2.imshow('teste', rgb)
        pass

    try:
        depth_array = next(depth_gen)
    except StopIteration:
        break
    else:
        # depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('teste2', depth)
        pass

    #depth = open3d.geometry.Image((z_norm * 255).astype(np.float32))
    #points3d = cv2.rgbd.depthTo3d(depth_array, cam_params.Kdepth)
    #color = open3d.geometry.Image(rgb)
    #depth = open3d.geometry.Image(depth_array)
    #camera_intrinsic = open3d.camera.PinholeCameraIntrinsic()
    #camera_intrinsic.intrinsic_matrix = cam_params.Kdepth
    #rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,convert_rgb_to_intensity=False)
    #pcd = open3d.geometry.RGBDImage.create_point_cloud_from_rgbd_image(rgbd, camera_intrinsic)

    # flip the orientation, so it looks upright, not upside-down
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #open3d.geometry.RGBDImage.draw_geometries([pcd])  # visualize the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xyz = get_3dPoints(depth_array,cam_params)
    ax.scatter(xyz[0], xyz[1], xyz[2])
    plt.show()
    rigid_transforms()
    cv2.waitKey(0)


print("Data saved on {}".format(args.transforms))

cv2.destroyAllWindows()
