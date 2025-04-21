# -*- coding: utf-8 -*-
# github: https://github.com/mattzheng/keras-yolov3-KF-objectTracking
'''
    File name         : kalman_filter.py
    File Description  : Kalman Filter Algorithm Implementation
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 3.6
    Fyi               : https://github.com/srianant/kalman_filter_multi_object_tracking
'''

# Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    This version tracks position (x, y) and velocity (vx, vy).
    """

    def __init__(self):
        """Initialize variables used by Kalman Filter class"""
        self.dt = 0.02  # delta time (20 ms)

        # State transition matrix (4x4)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt],
        ])

        # Observation matrix (2x4), we only observe position (x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Initial state [x, y, vx, vy]
        self.x = np.array([[-1], [-1], [0], [0]])
        self.u = np.zeros((2, 1))
        self.z = np.array([[-1], [-1]])  # measurement vector (position)

        # Covariance matrix (4x4)
        self.P = np.diag([3.0, 3.0, 3.0, 3.0])

        # Process noise covariance
        self.Q = np.diag([4.0, 4.0, 1.0, 1.0])

        # Measurement noise covariance
        self.R = np.diag([9.0, 9.0])

        self.prev_x = np.array([[-1], [-1], [0], [0]])
        self.prev_v = np.array([[0], [0]])

    def predict(self):
        """Predict the state and covariance matrix."""
        self.prev_x = self.x
        
        self.x = np.round(self.F @ self.x + self.B @ self.u, 2)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.T[0, :2]

    def correct(self, z, flag):
        """Update the Kalman Filter with new observation z.
        Args:
            z: measurement vector [x, y]
            flag: True if using actual detection, False if only prediction
        """
        
        self.z = z if flag else self.x[:2]

        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        y = self.z - self.H @ self.x  # Innovation
        self.x = np.round(self.x + K @ y, 2)  # Updated state estimate
        self.P = self.P - K @ S @ K.T  # Updated covariance estimate

        curr_v = (self.x[:2] - self.prev_x[:2]) / self.dt
        self.u = (curr_v - self.prev_v) / self.dt
        self.prev_v = curr_v

        return self.x.T[0, :2]


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount, position, label):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.alive = 0
        self.box = position
        self.label = label


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):  # (100, 8, 15, 100)
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh  # max distance: 100
        self.max_frames_to_skip = max_frames_to_skip  # trace max skip frames: 8
        self.max_trace_length = max_trace_length  # trace max history length: 15
        self.tracks = []
        self.delete_tracks_id = []
        self.trackIdCount = trackIdCount  # object id start from: 100

    def Update(self, detections, boxes):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount, boxes[i, :4], boxes[i, 4])
                track.KF.x = np.array([[detections[i][0]], [detections[i][1]], [0], [0]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                diff = self.tracks[i].prediction - detections[j]
                # print(f"after track[{i}].prediction: {self.tracks[i].prediction}, detections[{j}]: {detections[j]}, diff: {diff}")
                distance = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
                cost[i][j] = distance
        # print("\n")
        
        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = [-1] * N
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
            
        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                else:
                    self.tracks[i].box = boxes[assignment[i], :4]
                    self.tracks[i].label = boxes[assignment[i], 4]
                    self.tracks[i].alive += 1
            else:
                self.tracks[i].skipped_frames += 1


        # If tracks are not detected for long time, remove them
        del_tracks = [i for i in range(len(self.tracks)) if self.tracks[i].skipped_frames > self.max_frames_to_skip]
        self.delete_tracks_id = []
        
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks[::-1]:
                if id < len(self.tracks):
                    self.delete_tracks_id.append(self.tracks[id].track_id)
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = [i for i in range(len(detections)) if i not in assignment]

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in un_assigned_detects:
                track = Track(detections[i], self.trackIdCount, boxes[i, :4], boxes[i, 4])
                track.KF.x = np.array([[detections[i][0]], [detections[i][1]], [0], [0]])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            # print(f"predict state index {i} x: {self.tracks[i].KF.prev_x.T[0]} -> {self.tracks[i].KF.x.T[0]}, v: {self.tracks[i].KF.u.T[0]}")

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]][:, None], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.zeros((2, 1)), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            # print(f"correct state index {i} x: {self.tracks[i].KF.x.T[0]}")
            self.tracks[i].trace.append(self.tracks[i].prediction)
        
        # print("\n")
