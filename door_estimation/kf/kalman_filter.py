import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise as Q_noise

def kalman_filter(door_com_previous, door_com_measurement):
    # TODO: the filter trust the measurement too much
    dt = .1
    dim_x = 3
    dim_z = 3
    my_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    my_filter.x = door_com_previous          # initial state (location and velocity)

    my_filter.F = np.eye(dim_x)              # state transition matrix
    my_filter.H = np.eye(dim_x)              # Measurement function

    my_filter.P *= 1000.                     # covariance matrix
    my_filter.R *= 500.                      # state uncertainty
    my_filter.Q = Q_noise(3, dt, .1)         # process uncertainty

    my_filter.predict()
    my_filter.update(door_com_measurement)
    door_com_post = my_filter.x
    print('\n======== the posterior of door CoM position in World frame: \n', door_com_post)
    # sleep(.5)

    return door_com_post


## test data
# door_com_measurement = np.array([[13.], 
#                                 [21.], 
#                                 [0.]])
# door_com_previous = np.array( [[13.36229401],
#                             [21.65623708],
#                             [ 0.92096065]])
