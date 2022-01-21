import pyransac3d as pyrsc
import numpy as np
import cv2
import math  
import time
# import open3d as o3d
from pyobb.obb import OBB
import pcl

from skspatial.objects import Plane, Point
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise as Q_noise
from traitlets.traitlets import List


# colab scripts
def door_detection_filter(door_det, info):
    """
    description: Filter out the door candidates with wrong geometric characteristics.
    param:
        door_det: 
        info:
    return:
        door_det_filtered:
    """
    # x_thres = 0.1
    # y_thres = 0.4
    # xy_ratio_l = 0.30
    # xy_ratio_u = 1.0

    # # door width not too small
    # dx_rel = (door_det[:,2] - door_det[:,0]) / info[1]
    # # door height not too small
    # dy_rel = (door_det[:,3] - door_det[:,1]) / info[0]
    # # door width-height ratio in a proper range
    # xy_ratio = (door_det[:,2] - door_det[:,0]) / (door_det[:,3] - door_det[:,1])

    # logi_mat = np.stack((dx_rel>x_thres, dy_rel>y_thres, xy_ratio>xy_ratio_l, xy_ratio<xy_ratio_u), axis = 1)

    # red_logi_mat = np.logical_and.reduce(logi_mat, axis = 1)
    # door_det_filtered = door_det[red_logi_mat,:]

    # return door_det_filtered

def remove_outlier(pcd, nb_neighbors, std_ratio):
    """
    description:        Remove the outliers of the input pointcloud.
    param:
        pcd: 
        nb_neighbors: 
        std_ratio: 
    """
    # quit = False
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio) # ind: index
    # print('======== number of points after outlier removal: \n', len(ind))
    # if np.asarray(cl.points).shape[0] < 400:
    #     quit = True
    #     print('======== SKIP THE CURRENT FRAME BECAUSE NOT ENOUGH POINTS AFTER OUTLIER REMOVAL ========')

    # return cl, quit 
    
def obb(cl, door_plane):
    """
    description:    Apply a 3D bounding box to the input pointcloud after outlier removal, 
                    and project the box corners to the fitted plane of the door.
    param:
        cl:
        door_plane:
    """
    # bbox3d = o3d.geometry.OrientedBoundingBox.create_from_points(cl.points)
    # com_3d = bbox3d.center
    # hei_est = bbox3d.extent[0]
    # wid_est = bbox3d.extent[1]

    # pts8 = o3d.geometry.OrientedBoundingBox.get_box_points(bbox3d)
    # pts8 = np.asarray(pts8)

    # # get the 4 corners
    # pts4 = [(pts8[0]+pts8[3])/2, (pts8[1]+pts8[6])/2, \
    #           (pts8[2]+pts8[5])/2, (pts8[4]+pts8[7])/2]
    # pts4 = np.asarray(pts4)

    # # project the 4 points on the fitted plane
    # # convert Ax+By+Cz+D = 0 to z = ax+by+c
    # pts4_proj = []
    # for i in range(len(pts4)):
    #     point = Point(pts4[i])
    #     pts4_proj.append(door_plane.project_point(point))
    # pts4_proj = np.array(pts4_proj)

    # # 3D output
    # print('\n##### the estimated CoM position (mm): \n', com_3d)
    # print('\n##### the estimated door height (mm): \n', hei_est)
    # print('\n##### the estimated door width (mm): \n', wid_est)
    # print('\n##### the estimated 4 corner points in 3D (mm):\n', pts4_proj)

    # return pts4_proj, com_3d, wid_est

def back_projection(pts4_proj, com_3d):
    """
    description:    Project estimated 3D corner points and CoM point of the door back to 2D.
    param:
        pts4_proj:
        com_3d:
    """
    # img_path = '/content/drive/MyDrive/ETH/THESIS/estimation_pip/color_aligned/color_aligned.jpeg'
    # img_rgb_color = cv2.imread(img_path)
    # crs_2d = []
    # # u = X/Z*fx + cx
    # # v = Y/Z*fy + cy
    # for i in range(len(pts4_proj)):
    #     crs_2d.append([fx*pts4_proj[i][0]/pts4_proj[i][2]+cx, fy*pts4_proj[i][1]/pts4_proj[i][2]+cy])
    # crs_2d = np.array(crs_2d)
    # print('\n##### the corner points in 2D:\n', crs_2d)
    # com_2d = [fx*com_3d[0]/com_3d[2]+cx, fy*com_3d[1]/com_3d[2]+cy]   
    # print('\n##### the CoM point in 2D:\n', com_2d)

    # # copy the rgb image
    # img_rgb_cp = img_rgb_color

    # # corner points in integer
    # dt0 = (int(crs_2d[0][0]),int(crs_2d[0][1]))
    # dt1 = (int(crs_2d[1][0]),int(crs_2d[1][1]))
    # dt2 = (int(crs_2d[2][0]),int(crs_2d[2][1]))
    # dt3 = (int(crs_2d[3][0]),int(crs_2d[3][1]))

    # # draw circles and lines
    # cc = cv2.circle(img_rgb_cp,(int(com_2d[0]),int(com_2d[1])), 5, (0,255,0), 2)
    # for j in range(len(crs_2d)):
    #     cc = cv2.circle(img_rgb_cp,(int(crs_2d[j][0]),int(crs_2d[j][1])), 5, (0,255,0), 2)  
    # cc = cv2.line(img_rgb_cp, dt0, dt1, (0,0,255), 2)
    # cc = cv2.line(img_rgb_cp, dt0, dt2, (0,0,255), 2)
    # cc = cv2.line(img_rgb_cp, dt3, dt1, (0,0,255), 2)
    # cc = cv2.line(img_rgb_cp, dt3, dt2, (0,0,255), 2)

    # # visualisation
    # # cv2_imshow(cc)

def normal_cal(normal, normal_ini, i):
    """
    description: Calculate the door angle.
    param:
        pts4_proj:
        com_3d:
    """
    # angle = 0
    # if i == 1:
    #     normal_ini = normal
    # elif i > 1:
    #     dot_product = np.dot(normal_ini, normal)
    #     angle = np.arccos(dot_product)
    #     angle = angle*180/math.pi
    # normal_rounded = [round(normal[0],1), round(normal[1],1), round(normal[2],1)]
    # # text_display.append(round(wid_est*1000, 0))
    # text_display.append(normal_rounded)
    # text_display.append(round(angle, 2))
    # normals.append(normal)

def expand_bbox(
	go_door: bool, 
	class_det: list, 
	add: int, 
	w: int, 
	h: int
	):
	"""
	description:    Expand the bounding box in case the detected object is not fully captured.
	param:
		go_door:    
		class_det:  list[N,6], results of all yolo detections of a certain class, 
					[[result_boxes, result_scores, result_classid], [result_boxes, 
					result_scores, result_classid], ...]
		add:        int, an expansion factor to enlarge the bbox a bit
		w:          int, image width
		h:          int, image height

	return:
		xyxy:       tuple[4], result_boxes: xmin, ymin, xmax, ymax
		xcen:       float, the x coordinate of the box center
	"""
	# # expand the bbox more if going handle estimation
	# if go_door:  
	# 	add *= 4 
	
	# # pick the most confident detection in this class
	# class_best = confidence_filter(class_det)

	# # door bbox
	# xmin, ymin = int(class_best[0]), int(class_best[1])
	# xmax, ymax = int(class_best[2]), int(class_best[3])
	# xcen = (xmin+xmax) / 2.

	# # expand the bbox
	# xmin_, ymin_ = max(xmin-add, 0), max(ymin-add, 0)
	# xmax_, ymax_ = min(xmax+add, w), min(ymax+add, h)    

	# xyxy = (xmin_, ymin_, xmax_, ymax_)

	# return xyxy, xcen


# motpy tracking
def tracking(
    self
    ):
    """
    description:    	Track the objects of interest based on prior or new detections
    param:
        xyxy_door:      list[4], result_boxes: xmin, ymin, xmax, ymax
    return:
        xyxy_track:		list[N], all track boxes, e.g. [[xmin, ymin, xmax, ymax], 
                        [xmin, ymin, xmax, ymax], ...]
    """
    # update the state of the multi-object-tracker tracker
    if self.door_is_detected:
        self.tracker.step(detections=[Detection(box=self._xyxy_door)])
    if self.handle_is_detected:
        self.tracker.step(detections=[Detection(box=self._xyxy_handle)])

    # retrieve the active tracks from the tracker 
    tracks = self.tracker.active_tracks(max_staleness=5, min_steps_alive=3)

    # track results
    xyxy_track = []
    for i in range(len(tracks)):
        box = tracks[i].box
        xyxy_track.append([int(box[j]) for j in range(len(box))])

    return xyxy_track
 
def track_filter(
    self, 
    xyxy_track: list, 
    ):
    """
    description:    Get the 2D bbox (door or handle) and the door hinge side.
    param:
        det:        list[N,6], results of yolo detection, [[result_boxes, result_scores, 
                    result_classid], [result_boxes, result_scores, result_classid], ...]
    return:
        xyxy_door:       list[4], result_boxes: xmin, ymin, xmax, ymax
        xyxy_handle:       list[4], result_boxes: xmin, ymin, xmax, ymax
        hinge_side: str, the estimated hinge side: left, right, or unknown
    """

    xyxy_door_track = []
    xyxy_handle_track = []
    xyxy_4pcd = []
    self.c_handle = []
    door_ref = self.door_dim_ref
    handle_ref = self.handle_dim_ref

    # filter doors and handles track
    for i in range(len(xyxy_track)):
        # bbox dimension reference: height+width 
        dim = xyxy_track[i][2] - xyxy_track[i][0] + xyxy_track[i][3] - xyxy_track[i][1]
        # door filter
        if door_ref > .0:
            cond1 = dim/door_ref > .8
            cond2 = dim/door_ref < 1.2
            if len(xyxy_door_track) == 0 and cond1 and cond2:
                xyxy_door_track = xyxy_track[i]
                continue
        # handle filter
        if handle_ref > .0:
            cond1_ = dim/handle_ref > .8
            cond2_ = dim/handle_ref < 1.2
            if len(xyxy_handle_track) == 0 and cond1_ and cond2_:
                xyxy_handle_track = xyxy_track[i]

    # invalidate the handle bbox if it is outside the door bbox (tightening buffer: 2)
    if len(xyxy_door_track) > 0 and len(xyxy_handle_track) > 0:
        cond3 = xyxy_handle_track[0] < xyxy_door_track[0]+2
        cond4 = xyxy_handle_track[2] > xyxy_door_track[2]-2
        if cond3 and cond4:
            xyxy_handle_track = []

    self.door_is_tracked = len(xyxy_door_track) > 0
    self.handle_is_tracked = len(xyxy_handle_track) > 0

    # handle bbox center
    if self.handle_is_tracked:
        self.c_handle = [int(.5 * (xyxy_handle_track[2]+xyxy_handle_track[0])), 	
                        int(.5 * (xyxy_handle_track[3]+xyxy_handle_track[1]))]	
            
    # door pcd is prior to handle's
    if self.door_is_tracked:
        self.is_door_pcd = True
        xyxy_4pcd = xyxy_door_track
    elif self.handle_is_tracked:
        self.is_door_pcd = False
        xyxy_4pcd = xyxy_handle_track
    
    self.xyxy_handle_track = xyxy_handle_track
    self.xyxy_door_track = xyxy_door_track

    print('======== door bbox after track filter: ', xyxy_door_track)
    print('======== handle bbox after track filter: ', xyxy_handle_track)

    return xyxy_4pcd


def handle_estimate(self):
    """
    description: 	Estimate measurement of handle CoM position in Door and Camera frames
    """ 
    xmin_h_, ymin_h_, xmax_h_, ymax_h_ = [int(h) for h in self.xyxy_handle]
    # if self.height_is_valid:
    # 	## find handle CoM in Door by 2D bbox projection
    # 	# handle pos in Door keeps updating even if door detection is incomplete
    # 	xmin_d_, ymin_d_, xmax_d_, ymax_d_ = [int(d) for d in self.xyxy_door]

    # 	# find handle CoM position in Door frame
    # 	x_com_door = .5 * (xmin_d_+xmax_d_)
    # 	y_com_door = .5 * (ymin_d_+ymax_d_)
    # 	x_com_handle = .5 * (xmin_h_+xmax_h_)
    # 	y_com_handle = .5 * (ymin_h_+ymax_h_)
    # 	diff_x = x_com_handle - x_com_door
    # 	diff_y = y_com_handle - y_com_door
        
    # 	# width and height of door 2D bbox
    # 	w_door = xmax_d_ - xmin_d_
    # 	h_door = ymax_d_ - ymin_d_
        
    # 	# handle CoM in Door frame
    # 	h, w = self.door_height, self.door_width
    # 	x_handle_D = -diff_y / h_door * h
    # 	y_handle_D = -diff_x / w_door * w
    # 	self.handle_D = np.array([x_handle_D, y_handle_D, .0])

    # 	if self.do_vis:
    # 		# convert handle CoM from Door frame to World frame
    # 		trans_WD_T = self.trans_WD.reshape((3,1))
    # 		handle_D_T = self.handle_D.reshape((3,1))
    # 		handle_W_T = np.matmul(self.mat_WD, handle_D_T) + trans_WD_T	# [3,1]

    # 		# convert handle CoM from World frame to Camera frame
    # 		trans_CW = -np.matmul(self.mat_CW, self.trans_WC_T)
    # 		handle_C_T = np.matmul(self.mat_CW, handle_W_T) + trans_CW		# [3,1] 
    # 		self.handle_C = handle_C_T.T[0]									# [3,]


def kalman_filter(self) -> np.ndarray:
    """
    description:                A Kalman Filter that avgs door CoM position 
                                in the past estimation loops.
    param:
        door_com_previous:      np.ndarray[3,1], the prior
        trans_world2door:       np.ndarray[3,1], the measurement
    return:
        door_com_post:          np.ndarray[3,1], the posterior
    """
    # dt = .1
    # dim_x = 3
    # dim_z = 3
    # my_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # my_filter.x = self._door_com_kf         # initial state (location and velocity)

    # my_filter.F = np.eye(dim_x)             # state transition matrix
    # my_filter.H = np.eye(dim_x)             # Measurement function

    # my_filter.P *= 10.                      # covariance matrix
    # my_filter.R *= 50.                      # state uncertainty
    # my_filter.Q = Q_noise(3, dt, .1)        # process uncertainty

    # my_filter.predict()
    # my_filter.update(self.trans_world2door)
    # door_com_post = my_filter.x

# 7 KF averaging
def kalman_filter(self, m=.2, n=.8):
    """
    description:    A Kalman Filter that averages door CoM position 
                    in the past estimation loops.
    """
    ## 7.1 Luenberger observer (manual gain selection)
    # # TODO: parameter tuning
    # if self.valid_count == 0:
    # 	self.trans_WD_kf = self.trans_WD							# np.ndarray[3,]
    # 	self.quat_WD_kf = self.quat_WD								# np.ndarray[4,]
    # 	self.door_height_kf = self.door_height						# float
    # 	self.door_width_kf = self.door_width						# float
    # else:
    # 	# stop update if the door is no more complete in the view
    # 	if self.height_is_valid and self.width_is_valid:
    # 		self.trans_WD_kf = m*self.trans_WD + n*self.trans_WD_kf
        
    # 	# weighted mean of quaternion
    # 	self.q_mean = R.from_quat([self.quat_WD_kf, self.quat_WD])
    # 	self.quat_WD_kf = self.q_mean.mean([n, m]).as_quat()
        
    # 	if self.height_is_valid:
    # 		self.door_height_kf = m*self.door_height + n*self.door_height_kf
    # 	if self.width_is_valid:
    # 		self.door_width_kf = m*self.door_width + n*self.door_width_kf

    # # handle com position in Door frame
    # if self.handle_is_detected and self.handle_is_valid:
    # 	if self.handle_D_kf[0] == 0.:
    # 		self.handle_D_kf = self.handle_D				# np.ndarray[3,]
    # 	else:
    # 		self.handle_D_kf = m*self.handle_D + n*self.handle_D_kf

    # # update the rotation instances
    # self.q_WD_kf = R.from_quat(self.quat_WD_kf)
    # self.mat_WD_kf = self.q_WD_kf.as_matrix()