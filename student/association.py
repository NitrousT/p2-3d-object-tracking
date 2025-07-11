# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        self.association_matrix = np.zeros((len(track_list), len(meas_list)))
        self.unassigned_tracks = []
        self.unassigned_meas = []

        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                # Calculate Mahalanobis distance
                mhd = self.MHD(track, meas, KF)

                # Check if the measurement lies inside the track's gate
                if self.gating(mhd, meas.sensor):
                    # If it's within the gate, store the MHD in the association matrix
                    self.association_matrix[i, j] = mhd
                else:
                    self.association_matrix[i,j] = np.inf

        self.unassigned_tracks = list(range(len(track_list)))
        self.unassigned_meas = list(range(len(meas_list)))
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        update_track, update_meas = np.unravel_index(np.argmin(self.association_matrix), \
                                                     self.association_matrix.shape)
        if self.association_matrix[update_track, update_meas] == np.inf:
            return np.nan, np.nan

        track_idx, meas_idx = self.unassigned_tracks[update_track], self.unassigned_meas[update_meas]

        del self.unassigned_tracks[update_track]
        del self.unassigned_meas[update_meas]
                
        self.association_matrix = np.delete(self.association_matrix, update_track, axis=0)
        self.association_matrix = np.delete(self.association_matrix, update_meas, axis=1)
            
        ############
        # END student code
        ############ 
        return track_idx, meas_idx
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        gate_threshold = chi2.ppf(params.gating_threshold, df=sensor.dim_meas) #
        return MHD < gate_threshold    
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        try:
            gamma = KF.gamma(track, meas)
        except NameError: # Handle cases where get_hx might raise an error (e.g., division by zero in camera projection)
            return np.inf 
        
        # Calculate innovation covariance S = H * P * H^T + R
        H = meas.sensor.get_H(track.x)
        S = H @ track.P @ H.transpose() + meas.R #

        # Check for singular matrix (determinant close to zero)
        if np.linalg.det(S) == 0: #
            return np.inf #

        # Calculate Mahalanobis distance squared: gamma.transpose() * S.I * gamma
        MHD = float(gamma.transpose() @ np.linalg.inv(S) @ gamma) #
        
        return MHD #
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        for track in manager.track_list:          
            track.num_frames += 1
            
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:          
            print('track', track.id, 'score =', track.score)