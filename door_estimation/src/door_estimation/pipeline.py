from os import set_blocking
import numpy as np
import cv2
import time
import math
import pcl
import apriltag
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from yolov5_trt import run_yolov5, YoLov5TRT, plot_one_box

# 1.1
def confidence_filter(all_detections: list) -> list:
	"""
	description:        Select from candidate detections with the highest confidence
	param:
		all_detection:  list[N,6] (N>=0), results from Yolo detector output
	return:
		best_detection: list[n,6] (n=0 or n=1), the best result with the highest confidence
	"""
	if len(all_detections) == 0:
		return []
	elif len(all_detections) == 1:
		return all_detections[0]
	else:
		max_det_idx = np.argmax(all_detections, axis=0)
		max_conf_det = max_det_idx[4]
		best_detection = all_detections[max_conf_det] 

		return best_detection

# 1.2
def hinge_detection(
	door_det_best: list, 
	handle_det_best: list
	):
	"""
	description:            Determine the hinge side by the relative positions of 
							handle and door.
	param:
		handle_detection:   list[6], result_boxes, result_scores, result_classid
		door_center_x:      float, x coordinate of door CoM in the image plane
	return:
		hinge_side:         string, the hinge side (left, right or unknown)
	"""
	door_com_x = .5 * (door_det_best[0]+door_det_best[2])			# door CoM x position on image
	half_door_width = .5 * (door_det_best[2]-door_det_best[0])		# half door width on image
	handle_x = .5 * (handle_det_best[0]+handle_det_best[2])		# handle CoM x position on image
	diff_com_x = handle_x - door_com_x

	# determine the hinge side
	if diff_com_x > 0 and diff_com_x < half_door_width:
		hinge_side = 'left'
	elif diff_com_x < 0 and diff_com_x < half_door_width:
		hinge_side = 'right'
	else:
		hinge_side = 'unknown'
	
	# print('======== the estimated door hinge side: ', hinge_side)

	return hinge_side

# 3.1
def do_ransac_plane_segmentation(
	pcl_data,
	pcl_sac_model_plane,
	pcl_sac_ransac,
	max_distance
	):
	'''
	Create the segmentation object
	param:
		pcl_data: 				point could data subscriber
		pcl_sac_model_plane: 	use to determine plane models
		pcl_sac_ransac: 		RANdom SAmple Consensus
		max_distance: 			Max distance for apoint to be considered fitting the model
	return: 
		seg:					segmentation object
	'''
	seg = pcl_data.make_segmenter()
	seg.set_model_type(pcl_sac_model_plane)
	seg.set_method_type(pcl_sac_ransac)
	seg.set_distance_threshold(max_distance)
	return seg


class DoorEstimation(object):
	"""
	description:    This class wraps around the yolov5 detection model to perform inference
					and estimation of door and hanlde using ROS messages.
	"""   
	def __init__(
		self,
		intrinsics: list,
		info: list,
		yolov5_wrapper: YoLov5TRT,
		do_vis: bool = False,
		do_print: bool = False,
		do_selectROI: bool = False,
		run_apriltag: bool = False,
		run_csrt: bool = False,
		src_frame: str = 'world_corrected',
		):
		self.intri = np.array(intrinsics).reshape((3, 3))		# np.array(3,3), float
		self.info = info
		self.yolov5_wrapper = yolov5_wrapper
		self.src_frame = src_frame

		self.do_vis = do_vis
		self.do_print = do_print
		self.do_selectROI = do_selectROI
		self.run_apriltag = run_apriltag
		self.run_csrt = run_csrt

		self.h = info[0]
		self.w = info[1]

		self.loop_count = 0
		self.count_yolo = 0
		self.count_d = 0
		self.count_h = 0
		self.count_gt_pos = 0
		self.count_gt_nor = 0
		self.avg_t = 0

		self.t_total = []
		self.t_val = []
		self.t_infer = []
		self.t_yolo = []
		self.t_flt = []
		self.t_dsp = []
		self.t_rsc = []
		self.t_obb = []
		self.t_handle = []
		self.t_kf = []

		self.trans_WD_kf = np.zeros(3)
		self.normal_W_kf = np.zeros(3)
		self.door_height_kf = .0
		self.door_width_kf = .0
		self.handle_D_kf = np.zeros(3)
		self.handle_C_kf = np.zeros(3)
		self.door_position_gt_W_post = np.zeros(3)
		self.door_normal_gt_W_post = np.zeros(3)

		self.roi_is_selected = False

		# initialize commercial tracker
		# tracker_types = ['Boosting', 'MIL','KCF', 'TLD', 'MedianFlow', 'GOTURN', 'MOSSE', 'CSRT']
		self.csrt = cv2.TrackerMIL_create()

		# initialize Kalman Filter
		self.dim_x = 3+3+1+1+3
		dim_z = self.dim_x
		self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=dim_z)

	"""
	Operations
	""" 
	def csrt_launcher(
		self,
		) -> bool:
		
		frame = self.rgb_input
		
		# update tracker
		ok, bbox = self.csrt.update(frame)
		print('@@@@@@@csrt bbox: ', bbox)

		# plot
		bbox_xyxy = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
		if self.do_vis:
			plot_one_box(
				bbox_xyxy, self.rgb_csrt, color=(0,0,255), line_thickness=3
			)


	def read_apriltag(
		self,
		tag_size = 0.0808,
		):
		"""
		description:    The function is used to obtain the ground truth of door pose in W from
						Apriltags placed on the four corners of the target door.
		"""
		camera_intrinsics_vec = self.info[2:6]
		# convert RGB image to grayscale
		rgb_weights = [0.2989, 0.5870, 0.1140]
		grayscale_img = np.dot(self.rgb_input[...,:3], rgb_weights)
		grayscale_img = np.asarray(grayscale_img, dtype="uint8")
		options = apriltag.DetectorOptions(families='tag36h11',
										border=1,
										nthreads=4,
										quad_decimate=1.0,
										quad_blur=0.0,
										refine_edges=True,
										refine_decode=False,
										refine_pose=True,
										debug=False,
										quad_contours=True)
		detector = apriltag.Detector()
		detections = detector.detect(grayscale_img)
		detections_t = []
		detections_n = []
		tag_id = []

		for i in range(len(detections)):
			tag_id.append(detections[i].tag_id)
			detections_pose = detector.detection_pose(
				detection=detections[i], camera_params=camera_intrinsics_vec, 
				tag_size=tag_size, z_sign=1)
			# rot = detections_pose[0][0:3,0:3]
			normal = [detections_pose[0][0][2],detections_pose[0][1][2],detections_pose[0][2][2]]
			trans = [detections_pose[0][0][3],detections_pose[0][1][3],detections_pose[0][2][3]]
			detections_t.append(trans)
			detections_n.append(normal)

		if len(detections) > 0:
			# ground truth door normal in C
			door_normal_gt_C = (-sum(np.array(detections_n), 0)/len(detections)).tolist()

			# ground truth door CoM position in C
			if 0 in tag_id and 3 in tag_id:
				id0 = tag_id.index(0)
				id3 = tag_id.index(3)
				arr = np.array([detections_t[id0],detections_t[id3]])
				door_position_gt_C = (sum(arr,0)/2).tolist()
			elif 1 in tag_id and 2 in tag_id:
				id1 = tag_id.index(1)
				id2 = tag_id.index(2)
				arr = np.array([detections_t[id1],detections_t[id2]])
				door_position_gt_C = (sum(arr,0)/2).tolist()
			else:
				door_position_gt_C = []

			# convert from C to W
			# normal
			self.count_gt_nor += 1
			arr_n = np.reshape(np.array(door_normal_gt_C), (3,1))
			self.door_normal_gt_W = np.matmul(self.mat_WC, arr_n).T[0]			# [3,]
			if self.count_gt_nor == 1:
				self.door_normal_gt_W_post = self.door_normal_gt_W
			else:
				self.door_normal_gt_W_post = ((self.count_gt_nor-1)*self.door_normal_gt_W_post
					+self.door_normal_gt_W) / self.count_gt_nor			
			# position
			if len(door_position_gt_C) > 0:
				self.count_gt_pos += 1
				arr_t = np.reshape(np.array(door_position_gt_C), (3,1))
				self.door_position_gt_W = (np.matmul(self.mat_WC, arr_t) + self.trans_WC_T).T[0]	# [3,]
				if self.count_gt_pos == 1:
					self.door_position_gt_W_post = self.door_position_gt_W
				else:
					self.door_position_gt_W_post = ((self.count_gt_pos-1)*self.door_position_gt_W_post
						+self.door_position_gt_W) / self.count_gt_pos


	def validate_2dbbox(
		self,
		) -> bool:
		"""
		description:    validate the YOLO 2D bounding box by building a validation set
						from past estimation results in 3D and back projection to 2D.
		"""
		fx, fy, cx, cy = self.info[2:6]
		h = self.door_height_kf
		w = self.door_width_kf
		
		# convert from Door frame to World frame
		trans_WD_kf_stack = np.tile(self.trans_WD_kf.reshape((3,1)), 4)
		door_corners_D = np.array([	[h/2, h/2, -h/2, -h/2], 
									[w/2, -w/2, -w/2, w/2], 
									[.0, .0, .0, .0]])
		door_corners_W = np.matmul(self.mat_WD_kf, door_corners_D) + trans_WD_kf_stack

		# convert from World frame to Camera frame
		# be sure to use odometry in this loop
		stk = np.transpose(np.tile(self.trans_WC, (4,1)))
		trans_CW_stack = -np.matmul(self.mat_CW, stk)
		door_corners_C = \
			np.matmul(self.mat_CW, door_corners_W) + trans_CW_stack	# np.ndarray[3,4] 

		# handle conversion from W to C
		if self.count_h > 0:
			trans_CW = -np.matmul(self.mat_CW, self.trans_WC)
			self.handle_C_kf = np.matmul(self.mat_CW, self.handle_W_kf) + trans_CW

		# convert from Camera frame to Image frame
		cn = door_corners_C

		# rule out the multiple solutions when the camera coord. rotates 180 deg
		if False in list(cn[2]>0):
			self.xyxy_door_val = []
		else:
			door_corners_I = np.zeros((4,2))
			for i in range(4):
				door_corners_I[i][0] = int(fx*cn[0][i]/cn[2][i] + cx)
				door_corners_I[i][1] = int(fy*cn[1][i]/cn[2][i] + cy)
			
			# convert the quadrilateral in Image frame to rectangle
			sort1 = np.sort(door_corners_I[:,0])[::-1]	# x, from max to min
			sort2 = np.sort(door_corners_I[:,1])[::-1]	# y, from max to min
			min_x_v = int(.5 * (sort1[2]+sort1[3]))
			max_x_v = int(.5 * (sort1[0]+sort1[1]))
			min_y_v = int(.5 * (sort2[2]+sort2[3]))
			max_y_v = int(.5 * (sort2[0]+sort2[1]))
			if min_x_v > 0 and min_y_v > 0 and max_x_v < self.w and max_y_v < self.h:
				self.height_is_valid = True
				self.width_is_valid = True
			min_x = max(min_x_v, 0)
			max_x = min(max_x_v, self.w)
			min_y = max(min_y_v, 0)
			max_y = min(max_y_v , self.h)	
			self.xyxy_door_val = [min_x, min_y, max_x, max_y]

			if self.count_h > 0:
				# convert from Camera frame to Image frame
				hd = self.handle_C_kf
				x = int(fx*hd[0]/hd[2] + cx)
				y = int(fy*hd[1]/hd[2] + cy)
				self.handle_I_kf = np.array([x, y])

			if self.do_vis:
				plot_one_box(
					self.xyxy_door_val, self.rgb_val, color=(0,255,0), line_thickness=3
				)
				# rgb validation viz for posterior handle 
				if self.count_h > 0:
					cv2.circle(self.rgb_val, (x,y), 8, (255,0,0), 5)


	def detection_filter(
		self, 
		iou_thresh_door: float = .3,
		) -> bool:
		"""
		description:    Get the 2D bbox (door or handle) and the door hinge side.
		param:
			det:        list[N,6], results of yolo detection, [[result_boxes, result_scores, 
						result_classid], [result_boxes, result_scores, result_classid], ...]
		return:
			xyxy_door:       list[4], result_boxes: xmin, ymin, xmax, ymax
			xyxy_handle:     list[4], result_boxes: xmin, ymin, xmax, ymax
			hinge_side: str, the estimated hinge side: left, right, or unknown
		"""
		t0 = time.time()
		h, w = self.info[0], self.info[1]
		det = self.det

		det_d = det[det[:,5]!=1,:]
		det_h = det[det[:,5]==1,:]
		self.xyxy_door = []
		self.xyxy_handle = []
		
		# door filtration
		if len(det_d) > 0:
			if self.count_d > 0:
				# pick the max-iou door candidate from yolo
				x1, y1, x2, y2 = self.xyxy_door_val
				box_val = np.asarray(self.xyxy_door_val).reshape((1,4))
				box_val_a =  (x2-x1) * (y2-y1)
				if 0. < box_val_a < h*w:
					for i in range(len(det_d)):
						# calculate IOU of validation and yolo 2D bbox
						x1, y1, x2, y2 = det_d[i][0:4]
						box = np.asarray(det_d[i][0:4]).reshape((1,4))
						box_a =  (x2-x1) * (y2-y1)
						iou = YoLov5TRT.bbox_iou(self.yolov5_wrapper, box, box_val)
						# self.iou_stack.append(iou)
						# print('@@@@@@@avg iou: ', sum(self.iou_stack)/len(self.iou_stack))
						if iou >= iou_thresh_door and .7 < box_a / box_val_a < 1.3:
							self.xyxy_door = det_d[i][0:4]
							break
			else:
				det_d_best = confidence_filter(det_d)			# length of 0 or 6
				self.xyxy_door = det_d_best[0:4]
		self.door_is_detected = len(self.xyxy_door) > 0

		# handle filtration
		if len(det_h) > 0:
			box_d = []
			if self.door_is_detected:
				box_d = self.xyxy_door
			elif self.count_d > 0:
				box_d = self.xyxy_door_val				

			if len(box_d) > 0:
				# highest confident one within the 2D bbox
				if self.count_h == 0:
					x1, x2, y1, y2 = box_d[0], box_d[2], box_d[1], box_d[3]
					max_conf = 0.
					max_conf_i = -1
					
					for j in range(len(det_h)):
						x_h = int(.5 * (det_h[j][0]+det_h[j][2]))
						y_h = int(.5 * (det_h[j][1]+det_h[j][3]))
						cond1 = x_h > max(x1-5,0) and x_h < min(x2+5,w)
						cond2 = y_h > max(y1-5,0) and y_h < min(y2+5,h)
						if cond1 and cond2:
							if det_h[j][4] > max_conf:
								max_conf = det_h[j][4]
								max_conf_i = j
					
					if max_conf_i > -1:
						self.xyxy_handle = det_h[max_conf_i][0:4]					
				
				else:
					x1, x2, y1, y2 = box_d[0], box_d[2], box_d[1], box_d[3]
					x_h_val = self.handle_I_kf[0]
					y_h_val = self.handle_I_kf[1]

					if 0 < x_h_val < w and 0 < y_h_val < h:
						dis_thresh_h = 50.
						dis_min = 1e4
						dis_min_i = -1
						for j in range(len(det_h)):
							x_h = int(.5 * (det_h[j][0]+det_h[j][2]))
							y_h = int(.5 * (det_h[j][1]+det_h[j][3]))
							cond1 = x_h > max(x1-5,0) and x_h < min(x2+5,w)
							cond2 = y_h > max(y1-5,0) and y_h < min(y2+5,h)
							if cond1 and cond2:
								# closest one to the posterior handle
								dis = math.sqrt((x_h_val-x_h)**2+(y_h_val-y_h)**2)
								if dis < dis_min:
									dis_min = dis
									dis_min_i = j
						if dis_min_i > -1:
							self.xyxy_handle = det_h[dis_min_i][0:4]
		
		self.handle_is_detected = len(self.xyxy_handle) > 0
		t1 = time.time()
		self.t_flt.append(1e3*(t1-t0))

		if self.do_print:
			print('======== door bbox after yolo filter: ', self.xyxy_door)
			print('======== handle bbox after yolo filter: ', self.xyxy_handle)
		return self.door_is_detected or self.handle_is_detected
		

	def depth2pcd(
		self,
		leaf_size: float = .1,
		):
		"""
		description:        Convert the points inside the 2D bounding box of the depth 
							image to 3D points.
		"""
		xmin_d_, ymin_d_, xmax_d_, ymax_d_ = [int(d) for d in self.xyxy_door]
		
		# convert depth to pointcloud (H,W,3)
		self.pcd = cv2.rgbd.depthTo3d(self.dep_input, self.intri)	# [H,W,3], /mm

		# build pointcloud of the scene
		if self.do_vis:
			cp = np.copy(self.pcd)
			pcd_scene_T = cp.reshape(-1, 3)							# [H*W,3], /mm
			pcd_scene_T /= 1000
			data = pcl.PointCloud(pcd_scene_T.astype('f4'))
			leaf = data.make_voxel_grid_filter()
			leaf.set_leaf_size(leaf_size, leaf_size, leaf_size)
			pcl_ds = leaf.filter()				
			self.scene_pcd_C = pcl_ds.to_array() 					# /m

		# crop the pointcloud (h_box,w_box,3)
		door_crop = self.pcd[ymin_d_:ymax_d_, xmin_d_:xmax_d_, :]	
		# to (N,3)
		door_column = door_crop.reshape(-1, 3)
		# mm to m
		door_column /= 1000
		is_sparse =  door_column.shape[0] < 100
		self.door_column = door_column

		# print('======== Number of pcd after cropping: \t', self.door_column.shape[0])

		return is_sparse


	def pcd_processing(
		self,
		ds_voxel_size: float = .08,			
		ransac_thresh: float = .08,
		remove_outlier: bool = False,
		mean_k: int = 50,
		outlier_thresh: float = .15,
		):
		"""
		description: 		Downsample the pointcloud by a voxel grid filter. Fit a plane 
							of the input pointcloud and get the inliers. Remove the outliers.
		param:
			ds_voxel_size:  float, downsampling voxel(leaf) size
			mean_k:         int, number of neighboring points to analyze for any given point
			outlier_thresh: float, any point with a mean distance larger than this threshold 
							will be considered outlier

		return:
			inliers:    	np.ndarray[N,3], the pointcloud of inliers after RANSAC
			normal:     	tuple[3], a normalized normal vector of the fitted plane
			no_fit:     	bool, a flag of dropping this loop
		"""
		t0 = time.time()
		normal = (.0,.0,.0)
		xmin_d_, ymin_d_, xmax_d_, ymax_d_ = [int(d) for d in self.xyxy_door]
		# w_px = xmax_d_ - xmin_d_
		# h_px = ymax_d_ - ymin_d_

		# 3-1 downsample using voxel grid filter
		pcl_data = pcl.PointCloud(self.door_column.astype('f4'))
		vox = pcl_data.make_voxel_grid_filter()
		vox.set_leaf_size(ds_voxel_size, ds_voxel_size, ds_voxel_size)
		pcl_downsampled = vox.filter()
		
		if pcl_downsampled.width < 40:
			print('SKIP THE CURRENT FRAME BECAUSE TOO FEW POINTS AFTER DOWNSAMPLING')
			return False
			
		t1 = time.time()
		self.t_dsp.append(1000*(t1-t0))

		# 3-2 RANSAC Plane Segmentation
		ransac_segmentation = do_ransac_plane_segmentation(
			pcl_downsampled, pcl.SACMODEL_PLANE, pcl.SAC_RANSAC, ransac_thresh
		)
		# extract inliers after RANSAC, and coefficients of Ax+By+Cz+D=0
		inliers, cfc = ransac_segmentation.segment()
		pcl_inlier = pcl_downsampled.extract(inliers, negative=False)	# pcl.PointCloud

		if len(cfc) == 0:
			print('SKIP THE CURRENT FRAME BECAUSE NO FIT IN RANSAC')
			return False
		else:
			# obtain the normal of the fitted plane
			a = -cfc[0] / cfc[2]
			b = -cfc[1] / cfc[2]
			normal = (a, b, -1)
			normal /= np.linalg.norm(normal)
		
		t2 = time.time()
		self.t_rsc.append(1000*(t2-t1))

		# 3-3 outlier removal
		if remove_outlier:
			outlier_filter = pcl_inlier.make_statistical_outlier_filter()
			outlier_filter.set_mean_k(mean_k)
			outlier_filter.set_std_dev_mul_thresh(outlier_thresh)
			self.pcl_door_filtered = outlier_filter.filter()			# pcl.PointCloud
		else:
			self.pcl_door_filtered = pcl_inlier

		if self.pcl_door_filtered.width < 100:
			return False

		self.normal_C = normal
		print('#####door normal C: ', self.normal_C)

		if self.do_print:
			# print('======== pcd no. after downsampling: \t', pcl_downsampled.width)
			# print('======== pcd no. after RANSAC: \t', pcl_inlier.width)
			# print('======== the formulation of the fitted plane: \
			# 				\n\tz = ({:0.2f})*x+({:0.2f})*y+({:0.2f}) (m):'.format(a,b,c))
			# print('======== pcd no. after outlier removal: \t', self.pcl_door_filtered.width)
			print('======== [measurement] normal of the door in C: \t', normal)

		return True

	# 4
	def obb(self):
		"""
		description: Apply a 3D bounding box to the input pointcloud after outlier removal
		""" 
		t0 = time.time()
		engine = self.pcl_door_filtered.make_MomentOfInertiaEstimation()		
		engine.compute()
		
		[min_point, max_point, bbox_com, rot] = engine.get_OBB()
		self.door_pcd_C = self.pcl_door_filtered.to_array()			# np.nbdarray[N,3]

		# results of door state and parameters in Realsense coordinate
		self.trans_CD = bbox_com[0]										# list[3]
		extents = (max_point-min_point)[0]								# list[3]
		self.door_scale = extents
		self.door_width = extents[1]
		self.door_height = extents[0]

		if self.height_is_valid and self.width_is_valid:
			self.count_d += 1
		
		t1 = time.time()
		self.t_obb.append(1000*(t1-t0))


	def local2global(self):
		"""
		description:    Convert the door pose in local map (Realsense) to that 
						in global map (world) by introducing robot odometry.
		"""
		self.trans_CD_T = np.reshape(self.trans_CD, (3, 1))				# [3,1]
		normal_C_T = np.reshape(self.normal_C, (3, 1))					# [3,1]

		# translations from world to door
		tl =  np.matmul(self.mat_WC, self.trans_CD_T) + self.trans_WC_T
		self.trans_WD = tl.T[0]
		# normal in World frame
		# use normal instead of rotation matrix from OBB to avoid multiple expressions
		self.normal_W = np.matmul(self.mat_WC, normal_C_T).T[0]			# [3,]
		# the x, y axis of Door coordinate in World frame
		if self.src_frame == '/tag_1':
			x_D_in_W = np.array([0., 1., 0.])							# [3,]
		else:
			x_D_in_W = np.array([0., 0., 1.])							# [3,]
		z_D_in_W = self.normal_W										# [3,]
		y_D_in_W = np.cross(z_D_in_W, x_D_in_W)							# [3,]

		# handwritten rotation matrix from World to Door
		q = np.concatenate((x_D_in_W, y_D_in_W, z_D_in_W), axis=0)		# [9,]
		self.mat_WD = q.reshape((3,3)).T								# [3,3]
		self.q_WD = R.from_matrix(self.mat_WD)
		self.quat_WD = self.q_WD.as_quat()								# [4,]

		self.q_DC = self.q_WD.inv() * self.q_WC
		self.mat_DC = self.q_DC.as_matrix()
		self.mat_CD = self.mat_DC.T


	def handle_estimate(self):
		"""
		description: 	Estimate measurement of handle CoM position in Door and Camera frames
		"""
		t0 = time.time()
		xmin_h_, ymin_h_, xmax_h_, ymax_h_ = [int(h) for h in self.xyxy_handle]
		xcen_h_ = int(.5*(xmin_h_+xmax_h_))
		ycen_h_ = int(.5*(ymin_h_+ymax_h_))

		# mini handle bbox
		x0 = max(xcen_h_-2, 0)
		x1 = min(xcen_h_+2, self.w)
		y0 = max(ycen_h_-5, 0)
		y1 = min(ycen_h_+5, self.h)

		cp = np.copy(self.pcd)
		handle_crop = cp[y0:y1, x0:x1, :]	
		handle_crop = handle_crop.reshape(-1, 3)
		# remove rows with all zeroes
		handle_crop = handle_crop[~np.all(handle_crop == 0, axis=1)]
		m = np.mean(handle_crop, axis=0)
		cond1 = not np.isnan(m[0])
		if cond1:
			self.handle_C = m / 1000
		else:
			self.handle_C = np.zeros((1,3))
		
		if not (self.height_is_valid and self.width_is_valid):
			# use posterior if target door is not complete in the view
			self.q_DC = self.q_WD_kf.inv() * self.q_WC
			self.mat_DC = self.q_DC.as_matrix()
			# translations from C to D
			trans_WD_kf_T = self.trans_WD_kf.reshape((3,1))
			self.trans_CD_T =  np.matmul(self.mat_CW, trans_WD_kf_T) + self.trans_CW_T
		
		# convert from Camera to Door
		handle_C_T = self.handle_C.reshape((3,1))
		trans_DC_T = -np.matmul(self.mat_DC, self.trans_CD_T)
		handle_D = np.matmul(self.mat_DC, handle_C_T) + trans_DC_T
		self.handle_D = handle_D.T[0]
		# self.handle_D[2] = 0

		# assume that a handle lies within a given area of a door
		x_m,y_m = self.handle_D[0:2]
		cond2 = abs(self.handle_D[0]) < .40
		cond3 = .25 < abs(self.handle_D[1]) < .55
		if self.count_h == 0:
			self.handle_is_valid = cond1 and cond2 and cond3
		else:
			x_p,y_p = self.handle_D_kf[0:2]
			cond4 = math.sqrt((x_m-x_p)**2 + (y_m-y_p)**2) < .15
			self.handle_is_valid = cond1 and cond2 and cond3 and cond4

		if self.handle_is_valid:
			self.count_h += 1

		t1 = time.time()
		self.t_handle.append(1000*(t1-t0))


	def kalman_filter(self):
		"""
		description:    A Kalman Filter that process the state vector using 
						Ricatti equation with Kalman gain.	
						The measurement takes the value of posterior if it 
						is invalid or no measurement at all.
		"""
		t0 = time.time()
		if self.count_h == 0:
			# intialize the non-handle terms when no handle estimation
			if self.height_is_valid and self.width_is_valid:
				self.trans_WD_kf = self.trans_WD
			self.normal_W_kf = self.normal_W
			if self.height_is_valid:
				self.door_height_kf = self.door_height
			if self.width_is_valid:
				self.door_width_kf = self.door_width
			# update the rotation instances
			self.mat_WD_kf = self.mat_WD
		
		elif self.count_h == 1 and self.handle_is_valid:
			# when a new handle detection fills in
			self.handle_D_kf = self.handle_D
			door_size = np.asarray([self.door_height_kf, self.door_width_kf])
			self.x_init = np.concatenate((
				self.trans_WD_kf, self.normal_W_kf, door_size, self.handle_D_kf
			))
			# initialize Kalman Filter
			self.kf.x = self.x_init				# initial state (location and velocity)
			self.kf.F = np.eye(self.dim_x)		# state transition matrix
			self.kf.H = np.eye(self.dim_x)		# measurement matrix
			self.kf.P *= 50.					# covariance matrix
			self.kf.Q *= 1.  					# process uncertainty
			self.kf.R *= 100					# sensor measurement uncertainty
		
		else:
			# # varied measurement noise
			# cff = 8.
			# a,b,c = self.handle_C_kf
			# dis_CH = math.sqrt(a**2 + b**2 + c**2)
			# self.kf.R = cff * dis_CH * np.eye(self.dim_x) 

			# gather measurements
			z0 = self.trans_WD \
				if self.height_is_valid and self.width_is_valid else self.trans_WD_kf
			z1 = self.normal_W
			z2 = self.door_height if self.height_is_valid else self.door_height_kf
			z3 = self.door_width if self.width_is_valid else self.door_width_kf
			door_size = np.asarray([z2, z3])
			z4 = self.handle_D if self.handle_is_valid and self.width_is_valid else self.handle_D_kf
			z = np.concatenate((z0, z1, door_size, z4))

			# prediction update 
			self.kf.predict()
			# measurement update 
			self.kf.update(z)
			x_post = self.kf.x_post
			self.trans_WD_kf = x_post[0:3]
			self.normal_W_kf = x_post[3:6]
			self.door_height_kf = x_post[6]
			self.door_width_kf = x_post[7]
			self.handle_D_kf = x_post[8:11]

			# update the rotation instances
			x_D_in_W_kf = np.array([0., 1., 0.]) \
				if self.src_frame == 'tag_1' else np.array([0., 0., 1.])
			y_D_in_W_kf = np.cross(self.normal_W_kf, x_D_in_W_kf)
			q_kf = np.concatenate((x_D_in_W_kf, y_D_in_W_kf, self.normal_W_kf), axis=0)	
			self.mat_WD_kf = q_kf.reshape((3,3)).T

		# update quaternion
		self.q_WD_kf = R.from_matrix(self.mat_WD_kf)
		self.quat_WD_kf = self.q_WD_kf.as_quat()

		if self.handle_is_valid > 0:
			# convert handle CoM from Door frame to World frame
			trans_WD_kf_T = self.trans_WD_kf.reshape((3,1))
			handle_D_kf_T = self.handle_D_kf.reshape((3,1))
			handle_W_kf_T = np.matmul(self.mat_WD_kf, handle_D_kf_T) + trans_WD_kf_T	
			self.handle_W_kf = handle_W_kf_T.T[0]

		t1 = time.time()
		self.t_kf.append(1000*(t1-t0))

		
	def visualize(self): 
		"""
		description:	Visualize the estimation results in ROS
		"""
		fx, fy, cx, cy = self.info[2:6]

		# scatter image plot 
		# point cluster of the ROI
		if self.door_is_detected:
			pcd = self.door_pcd_C 
			for i in range(pcd.shape[0]):
				p = [fx*pcd[i][0]/pcd[i][2] + cx, fy*pcd[i][1]/pcd[i][2] + cy] 
				cv2.circle(self.rgb_scatter, (int(p[0]), int(p[1])), 3, (0,255,0), 1)
		# handle CoM
		if self.handle_is_valid:
			hd = self.handle_C
			hd_2d = [fx*hd[0]/hd[2] + cx, fy*hd[1]/hd[2] + cy] 
			cv2.circle(self.rgb_scatter, (int(hd_2d[0]), int(hd_2d[1])), 8, (255,0,0), 5)
	
		# scene pointcloud plot
		m = self.scene_pcd_C.shape[0]
		trans_WC_T_stack_ = np.tile(self.trans_WC_T, m)	# [3,n]
		scene_pcd_C_T = np.transpose(self.scene_pcd_C)	# [3,n]
		w = np.matmul(self.mat_WC, scene_pcd_C_T) + trans_WC_T_stack_
		self.scene_pcd_W = np.transpose(w)


	def printout(self):
		"""
		description:	Print out the estimation results
		"""
		m_val = sum(self.t_val)/len(self.t_val)
		m_yolo = sum(self.t_yolo)/len(self.t_yolo)
		m_infer = sum(self.t_infer)/len(self.t_infer)
		m_flt = sum(self.t_flt)/len(self.t_flt) if len(self.t_flt) > 0 else 0.
		m_dsp = sum(self.t_dsp)/len(self.t_dsp)
		m_rsc = sum(self.t_rsc)/len(self.t_rsc)
		m_obb = sum(self.t_obb)/len(self.t_obb)
		m_handle = sum(self.t_handle)/len(self.t_handle) if len(self.t_handle) > 0 else 0.
		m_kf = sum(self.t_kf)/len(self.t_kf) if len(self.t_kf) > 0 else 0.
		
		if self.height_is_valid:
			print('======== [measurement] Door height: {:0.3f}(m)'.format(self.door_height))
		if self.width_is_valid:
			print('======== [measurement] Door width: {:0.3f} (m)'.format(self.door_width))
		if self.height_is_valid and self.width_is_valid:
			print('======== [measurement] Door CoM in C: {}'.format(self.trans_CD))
		if self.handle_is_valid:
			print('======== [measurement] Handle position in C: ', self.handle_C)
			print('======== [measurement] Handle position in D: ', self.handle_D)

		print('======== [posterior] Door CoM in W:', np.around(np.transpose(self.trans_WD_kf),3))
		print('======== [posterior] Door normal in W: ', np.around(self.normal_W_kf,3))
		print('======== [posterior] Door height: {:0.3f}, width: {:0.3f}'.format(
						self.door_height_kf, self.door_width_kf
		))
		print('======== [posterior] Handle CoM in D: ', np.around(self.handle_D_kf,3))

		print('======== time total (ms):{:.1f} \n[validate]:{:.1f} \
			\n[yolo-inf]:{:.1f} \n[yolo-proc]:{:.1f} \n[inf-filter]:{:.1f} \
			\n[dsp]:{:.1f} \n[rsc]:{:.1f} \n[obb]:{:.1f}  \
			\n[handle est.]:{:.1f} \n[kf]:{:.1f}' \
			.format(self.m_total, m_val, m_infer, m_yolo-m_infer, m_flt, \
					m_dsp, m_rsc, m_obb, m_handle, m_kf))


	def estimate(
		self,
		rgb_input: np.ndarray,
		dep_input: np.ndarray,
		trans_WC: list,
		quat_WC: list,
		) -> bool:
		"""
		description:        Estimate the door and handle state and parameters.
		param:
			rgb_input:      np.ndaray[H,W,3], RGB image
			dep_input:    	np.ndaray[H,W], depth image 
			trans_WC: 		list[3], translation vector from world frame to Realsense 
							depth optical frame
			quat_WC:  		list[4], quaternion vector
			trans_WD_kf:	np.ndarray[3,1], the prior door CoM position
		return:
			is_estimated:	bool, if true, no failure of yolo or RANSAC, and thus 
							door or handle CoM is estimated
		"""
		self.rgb_input = rgb_input
		self.bgr_input = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2RGB)
		self.dep_input = dep_input
		self.trans_WC = trans_WC
		self.quat_WC = quat_WC

		# build transform instances
		self.q_WC = R.from_quat(self.quat_WC)			
		self.mat_WC = self.q_WC.as_matrix()
		self.mat_CW = self.q_WC.inv().as_matrix()
		self.trans_WC_T = np.reshape(self.trans_WC, (3, 1))		# [3,1]
		self.trans_CW_T = -np.matmul(self.mat_CW, self.trans_WC_T)
		self.xyxy_door_val = []

		# initialize flags
		self.is_detected = False
		self.door_is_detected = False
		self.handle_is_detected = False

		if self.count_d == 0:
			self.height_is_valid = True
			self.width_is_valid = True
		else:
			self.height_is_valid = False
			self.width_is_valid = False
		self.handle_is_valid = False
		self.loop_count += 1

		self.rgb_yolo = self.rgb_input.copy()
		self.rgb_val = self.rgb_input.copy()
		self.rgb_scatter = self.rgb_input.copy()
		if self.run_csrt:
			self.rgb_csrt = self.rgb_input.copy()

		if self.do_print:
			print('CALLBACK COUNT:          ', self.loop_count)
			print('YOLO INFERENCE COUNT:    ', self.count_yolo)
			print('DOOR ESTIMATION COUNT:   ', self.count_d)
			print('HANDLE ESTIMATION COUNT: ', self.count_h)

		start = time.time()
		if self.count_d > 0:
			self.validate_2dbbox()
			if len(self.xyxy_door_val) == 0:
				return False
		
		t0 = time.time()
		self.t_val.append(1e3*(t0-start))

		if self.run_apriltag:
			self.read_apriltag()

		self.det, self.rgb_yolo, self.is_detected, t = run_yolov5(
			self.rgb_yolo, self.yolov5_wrapper, self.do_vis
		) 

		self.t_infer.append(1e3*t)
		t1 = time.time()
		self.t_yolo.append(1e3*(t1-t0))

		if self.is_detected:
			self.count_yolo += 1

			if not self.roi_is_selected and self.do_selectROI:
				r = cv2.selectROI("Select the area of the target door please!", self.bgr_input)
				print('@@@@@@@@@@@ r: ', r)
				cv2.destroyAllWindows()
				self.xyxy_door = [int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3])]
				self.door_is_detected = True

				if self.run_csrt:
					frame = self.rgb_input
					ok = self.csrt.init(frame, r)
			else:
				go_on = self.detection_filter()
				if not go_on:
					return False
		else:
			return False

		if self.run_csrt:
			self.csrt_launcher()

		if self.door_is_detected:
			is_sparse = self.depth2pcd()
			if is_sparse:
				return False
			is_completed = self.pcd_processing()
			if not is_completed: 
				return False
			self.obb()
			self.local2global()

		if self.count_d > 0 and self.handle_is_detected:
			self.handle_estimate()

		if self.count_d > 0:
			self.kalman_filter()
		else:
			return False

		end = time.time()
		if self.count_d > 1:
			self.t_total.append(1e3*(end-start))

		if self.count_d == 1:
			self.roi_is_selected = True
		if self.do_vis:
			self.visualize()
			
		self.m_total = sum(self.t_total)/len(self.t_total) if len(self.t_total) > 0 else 0.
		if self.do_print:
			self.printout()
		
		return True
