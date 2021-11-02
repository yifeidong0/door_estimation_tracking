import numpy as np                       
import cv2                
import time

from collections import Counter
from PIL import Image
from pyobb.obb import OBB

from yolov5.tensorrt.yolov5_trt import main_yolov5
from estimation.hinge_side_detection import target_2dbbox
from estimation.candidate_pcd_3d import pcd_bbox_3d
from estimation.ransac import get_inl_ransac
from kf.door_position_rs_to_world import door_position_rs_to_world
from kf.kalman_filter import kalman_filter

## parameters
# acquisition of pcd
conf_thresh, iou_thresh = 0.1, 0.5
bbox_expand = 5
pcd3d_ds_rate_dr, pcd3d_ds_rate_hd = 8, 4
# RANSAC
ran_thresh_dr, ran_thresh_hd = 10, 14
ran_minPoints_dr, ran_minPoints_hd = 500, 500
# outlier removal
nb_neighbors, std_ratio = 10, 0.01
# record
angle, normals, normal_ini, hinge_sds, door_width_est,  = 0, [], 0.0, [], []
avg_time, valid_pcd_type, door_com_in_world_buf = [], [], []

def main_estimation(rgb_array, depth_array, items, 
                    rs_odom_trans, rs_odom_rot, count, door_prev, yolov5_wrapper):
    quit, text_display = False, []
    elapsed_time: str = 0.0
    time_0 = time.time()

    # (0) run yolov5-tenssorrt
    det, flag = main_yolov5(rgb_array, yolov5_wrapper, conf_thresh, iou_thresh) 
    if not flag:
        print('\n======== No DETECTION IN THE FRAME! ========\n')
        quit = True
        return door_prev, quit, elapsed_time
    time_1 = time.time()

    # (1) extract the most confident yolo detections and detect the hinge side 
    xminl, yminl, xmaxl, ymaxl, hinge_sd, quit, class_name = target_2dbbox(det, bbox_expand, items)
    # text_display.append(hinge_sd)
    if hinge_sd != 'unknown':
        hinge_sds.append(hinge_sd)
    if len(hinge_sds) > 0:
        occurence_count = Counter(hinge_sds)
        hinge_sd_pre = occurence_count.most_common(1)[0][0]
    else: 
        hinge_sd_pre = 'unknown'
    if quit: # continue if no door or handle detections
        return door_prev, quit, elapsed_time
    time_2 = time.time()

    # (2) get the 3D points inside the 2D bbox
    if class_name == 'Door': 
        in_bbox_3d, quit = pcd_bbox_3d(xminl, yminl, xmaxl, ymaxl, 
                    pcd3d_ds_rate_dr, hinge_sd_pre, depth_array, class_name, items)
    else:
        in_bbox_3d, quit = pcd_bbox_3d(xminl, yminl, xmaxl, ymaxl, 
                    pcd3d_ds_rate_hd, hinge_sd_pre, depth_array, class_name, items)
    if quit: 
        return door_prev, quit, elapsed_time
    # np.savetxt("in_bbox_3d.txt", in_bbox_3d)
    time_3 = time.time()

    # (3) RANSAC and get the inliers
    if class_name == 'Door': 
        inliers, fitted_plane, normal, quit = get_inl_ransac(
                            in_bbox_3d, ran_thresh_dr, ran_minPoints_dr)
    else: # this loop stops here if pcd is given by 'Handle'
        inliers, fitted_plane, normal, quit = get_inl_ransac(
                            in_bbox_3d, ran_thresh_hd, ran_minPoints_hd)
        normals.append(normal)
        return door_prev, True, elapsed_time
    if quit: 
        return door_prev, quit, elapsed_time
    # normal_cal(normal, normal_ini, i)
    time_4 = time.time()

    # (4) fit an oriented bounding box and back project to 2D
    obb = OBB.build_from_points(inliers)
    print('\n======== door CoM position:\n {}, \n\textents: \n{}'.format(obb.centroid, obb.extents))
    door_com_in_rs = obb.centroid
    rgb_array_cp = rgb_array
    com_2d = [items[2]*door_com_in_rs[0]/door_com_in_rs[2] + items[4], 
                items[3]*door_com_in_rs[1]/door_com_in_rs[2] + items[5]]   
    cc = cv2.circle(rgb_array_cp, (int(com_2d[0]), int(com_2d[1])), 
                    5, (0,255,0), 2)
    bgr_array = cv2.cvtColor(rgb_array_cp, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/reproj.jpg', bgr_array)
    time_5 = time.time()

    # (5) calculate door CoM position in World frame
    door_com_in_world = door_position_rs_to_world(rs_odom_trans, 
                                            rs_odom_rot, door_com_in_rs)
    time_6 = time.time()

    # (6) implement kalman filter
    if count == 0:
        door_this = door_com_in_world
    else:
        door_com_post = kalman_filter(door_prev, door_com_in_world)
        door_this = door_com_post
    
    door_com_in_world_buf.append(door_this)
    time_7 = time.time()

    elapsed_time = time_7-time_0
    print('\n======== time interval of all: {} (sec) \t(0): {} (sec) \
         \t(1): {} (sec) \t(2): {} (sec) \t(3): {} (sec) \t(4): {} (sec) \
          \t(5): {} (sec) \t(6): {} (sec)\n'.format( \
        time_7-time_0, time_1-time_0, time_2-time_1, time_3-time_2, 
        time_4-time_3, time_5-time_4, time_6-time_5, time_7-time_6))

    # normals = np.asarray(normals)
    # door_width_est = np.asarray(door_width_est)
    # avg_time = np.asarray(avg_time)
    # print('************** avg_time: ', np.mean(avg_time))
    # print('************** normals: ', normals)
    # print('************** avg normals: ', np.mean(normals, 0))
    # print('************** door_width_est: ', door_width_est)
    # print('************** avg door_width_est: ', np.mean(door_width_est))
    # print('************** estimated hinge sides: ', hinge_sds)
    # print('************** Filtered door CoM position (m): ', door_com_in_world_buf)

    return door_this, quit, elapsed_time

# def normal_cal(normal, normal_ini, i):
#     angle = 0
#     if i == 1:
#         normal_ini = normal
#     elif i > 1:
#         dot_product = np.dot(normal_ini, normal)
#         angle = np.arccos(dot_product)
#         angle = angle*180/math.pi
#     normal_rounded = [round(normal[0],1), round(normal[1],1), round(normal[2],1)]
#     # text_display.append(round(wid_est*1000, 0))
#     text_display.append(normal_rounded)
#     text_display.append(round(angle, 2))
#     normals.append(normal)
