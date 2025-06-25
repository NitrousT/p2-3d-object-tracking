# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
    # Track which detections have already been matched to a label
    matched_detection_indices = set()
    
    true_positives_count = 0 # This will be the count of correctly detected objects (labels that found a match)
    center_devs = []
    ious = []

    # Loop through each ground truth label
    for label_idx, (label, valid) in enumerate(zip(labels, labels_valid)):
        if not valid:
            continue # Skip invalid labels

        # print(f"\n--- Processing Label {label_idx} ---")
        # print(f"Label box center: ({label.box.center_x}, {label.box.center_y}, {label.box.center_z})")
        box_label = tools.compute_box_corners(label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.box.heading)
        # print(f"Label box corners: {box_label}") # Verify these look correct

        best_iou_for_label = -1
        best_match_for_label = None
        best_det_idx = -1

        # Loop over all detections to find the best match for the current label
        for det_idx, det in enumerate(detections):
            # Only consider detections that haven't been matched yet
            # This is important to prevent one detection from matching multiple labels
            if det_idx in matched_detection_indices:
                continue

            ## step 1 : extract the four corners of the current label bounding-box
            box = label.box
            box_label = tools.compute_box_corners(label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.box.heading)
            # print('  Label box w and l:',label.box.width, label.box.length ) 
            
            ## step 2 : extract the four corners of the current detection
            cls_id, x_world, y_world, z_world, h, w, l, yaw = det
            x_det_float = x_world.item() if hasattr(x_world, 'item') else x_world
            y_det_float = y_world.item() if hasattr(y_world, 'item') else y_world
            w_det_float = w.item() if hasattr(w, 'item') else w
            l_det_float = l.item() if hasattr(l, 'item') else l
            yaw_det_float = yaw.item() if hasattr(yaw, 'item') else yaw

            
            box_det = tools.compute_box_corners(x_det_float, y_det_float, w_det_float, l_det_float, yaw_det_float)
            # print(f" Det box w and l", w_det_float, l_det_float) 
            

            
            ## step 3 : compute the center distance between label and detection bounding-box in x, y, and z
            dist_x = np.array(box.center_x - x_world).item()
            dist_y = np.array(box.center_y - y_world).item()
            dist_z = np.array(box.center_z - z_world).item()
            
            ## step 4 : compute the intersection over union (IOU) between label and detection bounding-box
            current_iou = 0.0
            try:
                poly_1 = Polygon(box_label)
                poly_2 = Polygon(box_det)
                intersection = poly_1.intersection(poly_2).area 
                union = poly_1.union(poly_2).area
                if union > 0: # Avoid division by zero
                    current_iou = intersection / union
                # print(f"  Det {det_idx} - Intersection: {intersection:.4f}, Union: {union:.4f}, IOU: {current_iou:.4f}")
            except Exception as err:
                print(f"Error in IOU computation for label {label_idx} and detection {det_idx}: {err}")
            
            ## step 5 : if IOU exceeds min_iou threshold and is better than current best for this label
            if current_iou > min_iou and current_iou > best_iou_for_label:
                # print(f"    -> MATCH FOUND! Current IOU {current_iou:.4f} > min_iou {min_iou} AND better than best_iou {best_iou_for_label}")
                best_iou_for_label = current_iou
                best_match_for_label = [current_iou, dist_x, dist_y, dist_z]
                best_det_idx = det_idx
            else:
                print(f"    -> No match for det {det_idx}. IOU: {current_iou:.4f} (min_iou: {min_iou})")
        # After checking all detections for the current label, if a best match was found
        if best_match_for_label is not None:
            # print(f"Label {label_idx} found a best match with IOU: {best_match_for_label[0]:.4f}. Adding to TP.")
            true_positives_count += 1
            ious.append(best_match_for_label[0])
            center_devs.append(best_match_for_label[1:])
            # Mark the detection as used
            matched_detection_indices.add(best_det_idx)
        else:
            print(f"Label {label_idx} did NOT find a match.")

    # print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = labels_valid.sum()

    ## step 2 : compute the number of false negatives
    # False negatives are valid labels that did not find a matching detection
    false_negatives = all_positives - true_positives_count

    ## step 3 : compute the number of false positives
    # False positives are detections that did not match any valid label
    false_positives = len(detections) - true_positives_count
    
    # print(f"\n--- Final Metrics ---")
    # print(f"True Positives (TP): {true_positives_count}")
    # print(f"All Positives (total valid labels): {all_positives}")
    # print(f"False Negatives (FN): {false_negatives}")
    # print(f"False Positives (FP): {false_positives}")
    # print(f"Total Detections: {len(detections)}")
    # print(f"Length of ious list: {len(ious)}")
    # print(f"Length of center_devs list: {len(center_devs)}")

    pos_negs = [all_positives, true_positives_count, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    # print('IOUs:', ious, ' center_devs: ',center_devs)
    
    ####### ID_S4_EX3 START #######     
    #######    
    # print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    all_positives = sum(item[0] for item in pos_negs)
    true_positives = sum(item[1] for item in pos_negs)
    false_negatives = sum(item[2] for item in pos_negs)
    false_positives = sum(item[3] for item in pos_negs)
    
    ## step 2 : compute precision
    precision = true_positives /float(true_positives + false_positives)

    ## step 3 : compute recall 
    recall = true_positives / float(true_positives + false_negatives)

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

