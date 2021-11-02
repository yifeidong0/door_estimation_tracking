from scipy.spatial.transform import Rotation as R
import numpy as np

def door_position_rs_to_world(rs_trans, rs_quat, door_com_in_rs):

    rs_trans = np.asarray(rs_trans)
    rs_trans_t = np.reshape(rs_trans, (3, 1))
    rs_quat = np.asarray(rs_quat)

    r = R.from_quat(rs_quat)
    rot_world_to_rs = r.as_matrix()

    door_com_in_rs = np.array([ 674.02318455, 33.16628952, 5634.95259066])
    door_com_in_rs_t = np.reshape(door_com_in_rs, (3, 1))
    door_com_in_rs_t *= 0.001

    door_com_in_world = np.matmul(rot_world_to_rs, door_com_in_rs_t) + rs_trans_t
    print('\n======== RS position in World frame: \n', rs_trans_t)
    print('======== door CoM position in RS frame: \n', door_com_in_rs_t)
    print('======== door CoM position in World frame: \n', door_com_in_world)

    return door_com_in_world



## test data
# quat_test = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
# rs_trans = np.array([7.6884821940897465, 21.75736289536196, 0.8452569033226865])
# rs_quat = np.array([-0.5084514660584992, 0.47891962352766343, 
#                 -0.4688520075450197, 0.5406393399338313])
# door_normal = np.array([-0.16966488, 0.01462908, -0.98539323])