# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        F = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        return F
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        dt = params.dt
        q_cv = params.q # Process noise standard deviation for constant velocity

        Q = np.array([[0.25*dt**4, 0, 0, 0.5*dt**3, 0, 0],
        [0, 0.25*dt**4, 0, 0, 0.5*dt**3, 0],
        [0, 0, 0.25*dt**4, 0, 0, 0.5*dt**3],
        [0.5*dt**3, 0, 0, dt**2, 0, 0],
        [0, 0.5*dt**3, 0, 0, dt**2, 0],
        [0, 0, 0.5*dt**3, 0, 0, dt**2]]) * q_cv**2
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        Q = self.Q()

        x = track.x
        P = track.P

        # Predict state
        x_prime = F @ x

        # Predict covariance
        P_prime = F @ P @ np.transpose(F) + Q

        track.set_x(x_prime)
        track.set_P(P_prime)
        # print(f"DEBUG: Track ID {track.id} PREDICTED. New x: {track.x.T.A1}, New P_diag: {np.diag(track.P)}")
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        x = track.x # Predicted state from the track
        P = track.P # Predicted covariance from the track

        # Get Jacobian H from the measurement model using meas.sensor.get_H()
        H = meas.sensor.get_H(x) 

        # Calculate residual gamma
        gamma = self.gamma(track, meas)
        
        # Calculate covariance of residual S
        S = self.S(track, meas, H)
        
        # Calculate Kalman Gain (K)
        K = P @ H.T @ S.I # Use .I for matrix inverse for numpy.matrix
        
        # Update state
        x_updated = x + K @ gamma
        
        # Update covariance
        I = np.identity(P.shape[0]) # Identity matrix of the same size as P
        P_updated = (I - K @ H) @ P
        
        # print(f"DEBUG: Track ID {track.id} UPDATE step: gamma={gamma.T.A1}, S_diag={np.diag(S).T.A1}, K_diag={np.diag(K).T.A1}")
        track.set_x(x_updated)
        track.set_P(P_updated)
        # print(f"DEBUG: Track ID {track.id} UPDATED. Final x: {track.x.T.A1}, Final P_diag: {np.diag(track.P)}")
        
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        
        hx = meas.sensor.get_hx(track.x)
        gamma = np.matrix(meas.z - hx)
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        P = track.P
        R = meas.R

        S = H @ P @ np.transpose(H) + R

        return S
        
        ############
        # END student code
        ############ 