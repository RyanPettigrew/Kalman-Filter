import numpy as np

# Kalman Filter v1.0 for Flight Software Library

# [x, y, z] = position, [dx, dy, dz] = velocity
class KalmanFilter(object):
    def __init__(self, dt, accelVariance):
    
        self.dt = dt
        
        self.F = np.array([ [1, 0, 0, dt, 0, 0],
                            [0, 1, 0, 0, dt, 0],
                            [0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        self.H = np.array([ [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        self.P = np.eye(6)

        self.Q = accelVariance * np.array([ [(dt**4)/4, 0, 0, (dt**3)/2, 0, 0],
                                            [0, (dt**4)/4, 0, 0, (dt**3)/2, 0],
                                            [0, 0, (dt**4)/4, 0, 0, (dt**3)/2],
                                            [(dt**3)/2, 0, 0, dt**2, 0, 0],
                                            [0, (dt**3)/2, 0, 0, dt**2, 0],
                                            [0, 0, (dt**3)/2, 0, 0, dt**2]])

        #Measured covariance
        self.R = np.eye(6)
        self.x = np.array([[0], [0], [0], [0], [0], [0]])

    #Predict next state and covariance
    def predict(self):

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    #Calculate the Kalman gain and update the state estimates and covar
    def update(self, z):

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        return self.x
