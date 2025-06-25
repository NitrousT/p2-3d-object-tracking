# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        z_hom = np.ones((4,1))
        z_hom[:3] = meas.z # Only use x, y, z from measurement for initialization

        veh_pos = meas.sensor.sens_to_veh @ z_hom

        self.x = np.matrix([[veh_pos[0, 0]],
                        [ veh_pos[1,0]],
                        [ veh_pos[2,0]],
                        [ 0.        ],
                        [ 0.        ],
                        [ 0.        ]])


        P_pos = M_rot @ meas.R @ M_rot.transpose()

        P_vel = np.matrix([
            [params.sigma_p44**2, 0, 0],
            [0, params.sigma_p55**2, 0],
            [0, 0, params.sigma_p66**2]
        ])
        self.P = np.zeros((6, 6))
        self.P[:3,:3] = P_pos
        self.P[3:,3:] = P_vel
        
        self.state = 'initialized' # initial state
        # Initialize track score and history
        self.score = 1.0 / params.window # Initial score for a newly created track
        self.num_frames = 1

        # NEW: Flag to indicate if track was updated in current frame by ANY sensor
        self.is_updated_in_current_frame = True

         ############
        # END student code
        ############ 

        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t 

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates


class Trackmanagement:
    '''Track management class'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ############
        # TODO Step 2: implement track management for unassigned tracks:
        # - decrease track score for unassigned tracks
        # - delete old tracks
        # - initialize new track with unassigned measurement
        ############

        # Decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    track.score -= 1/params.window

        # delete old tracks   
        tracks_to_delete = []
        for track in self.track_list:
            if (track.state == 'confirmed' and track.score < params.delete_threshold) or\
                (track.state == 'initialized' and track.score <0.25 and track.num_frames>params.max_frames_assigned) or\
                (track.state == 'tentative' and track.score < 0.3):

                tracks_to_delete.append(track)


        for track in tracks_to_delete:
            self.delete_track(track)
            
        # initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        # A track being updated means it was associated with a measurement in this frame.
        track.score += 1/params.window
        track.score = min(track.score, 1.0)
        if track.state == "initialized" and track.score > params.tentative_threshold:
            track.state = "tentative"
            
        if track.state == "tentative" and track.score > params.confirmed_threshold:
            track.state = "confirmed"
        
        ############
        # END student code
        ############