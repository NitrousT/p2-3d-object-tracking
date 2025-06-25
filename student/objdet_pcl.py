# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d as o3d
import zlib
import tensorflow as tf

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Visualization", width=1280, height=720)
    
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)

    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    vis.run()
    vis.register_key_callback(262, lambda vis: vis.close())
    vis.poll_events()
    vis.update_renderer()
    vis.destroy_window()

    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
        H,W,C = ri.shape
        print(f"Range image shape: {H}x{W}x{C}")
        mid = W//2
        half_crop = W//4
        ri = ri[:,mid-half_crop:mid+half_crop,:]  # crop range image to center part for better visualization [+90 to -90 degrees]
    
    # step 2 : extract the range and the intensity channel from the range image
    range_channel = ri[:, :, 0]
    intensity_channel = ri[:, :, 1]
    
    # step 3 : set values <0 to zero
    range_channel[range_channel < 0] = 0.0
    intensity_channel[intensity_channel < 0] = 0.0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range_channel = (range_channel - np.min(range_channel)) / (np.max(range_channel) - np.min(range_channel)) * 255
    range_channel = range_channel.astype(np.uint8)
    
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    p1 = np.percentile(intensity_channel, 1)
    p99 = np.percentile(intensity_channel, 99)
    diff= p99 - p1

    if diff > 1e-5:
        intensity_channel = (intensity_channel - p1) / diff * 255
    else:
        intensity_channel = np.zeros_like(intensity_channel)  # or use default normalization

    intensity_channel = intensity_channel.astype(np.uint8)
    
    
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((range_channel, intensity_channel)).astype(np.uint8)
    
    # img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # print("\n--- DEBUG: bev_from_pcl configs ---")
    # print(f"configs.lim_x: {configs.lim_x}")
    # print(f"configs.lim_y: {configs.lim_y}")
    # print(f"configs.bev_width: {configs.bev_width}")
    # print(f"configs.bev_height: {configs.bev_height}")
    # bev_x_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    # bev_y_discret = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width
    # print(f"Calculated bev_x_discret (m/px for BEV Y): {bev_x_discret}")
    # print(f"Calculated bev_y_discret (m/px for BEV X): {bev_y_discret}")
    # print("------------------------------------\n")
    

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_x_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    bev_y_discret = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates   
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.floor((lidar_pcl_cpy[:, 0] - configs.lim_x[0]) / bev_x_discret).astype(np.uint16)
    
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.floor((lidar_pcl_cpy[:, 1] - configs.lim_y[0]) / bev_y_discret).astype(np.uint16)
    lidar_pcl_cpy[:, 1] = np.clip(lidar_pcl_cpy[:, 1], 0, configs.bev_width - 1)
    lidar_pcl_cpy[:, 0] = np.clip(lidar_pcl_cpy[:, 0], 0, configs.bev_height - 1)
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # show_pcl(lidar_pcl_cpy)
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width ))
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    
    lidar_sorted = lidar_pcl_cpy[np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, unique_indices, counts = np.unique(lidar_sorted[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_sorted[unique_indices]
    


    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    p1, p99 = np.percentile(lidar_pcl_top[:, 3], [1, 99])
    intensity = np.clip((lidar_pcl_top[:, 3] - p1) / (p99 - p1), 0, 1)

    row_indices = lidar_pcl_top[:, 0].astype(np.int32)
    col_indices = lidar_pcl_top[:, 1].astype(np.int32)
    intensity_map[row_indices, col_indices] = intensity
    
 
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # cv2.imshow("BEV Intensity Map", (intensity_map * 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height, configs.bev_width))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = \
        (lidar_pcl_top[:, 2] - configs.lim_z[0]) / (configs.lim_z[1] - configs.lim_z[0])
    height_map = np.clip(height_map, 0, 1)  # clip height values to valid range

    # step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    # cv2.imshow("Height Map", (height_map * 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #######
    ####### ID_S2_EX3 END #######       

    # # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    # lidar_pcl_top = []
    # height_map = []
    # intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height, configs.bev_width), dtype=np.float32)
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    xy_indices = lidar_pcl_cpy[np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)[1]][:, 0:2].astype(np.int32)
    normalized_counts = np.clip(np.log(counts + 1) / np.log(64), 0, 1)
    density_map[xy_indices[:, 0], xy_indices[:, 1]] = normalized_counts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width), dtype=np.float32)
    bev_map[0, :, :] = intensity_map
    bev_map[1, :, :] = height_map
    bev_map[2, :, :] = density_map

    # expand dimension of bev_map before converting into a tensor
    bev_maps = np.expand_dims(bev_map, axis=0)  # shape: [1, 3, H, W]
    bev_maps = torch.from_numpy(bev_maps).to(configs.device, non_blocking=True).float()
    
    return bev_maps
    


