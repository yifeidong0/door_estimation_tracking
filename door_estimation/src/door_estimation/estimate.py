"""
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
"""
import yaml
import rospy
import numpy as np
import tf
import math
import ctypes
import pycuda.driver as cuda
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from door_estimation_msgs.msg import DoorEstimationResult
from yolov5_trt import YoLov5TRT
from ros_utils import build_image_msg, read_image_msg, build_pointcloud_msg
from pipeline import DoorEstimation

with open('door_estimation/src/door_estimation/config.yaml') as f:
	cfg = yaml.load(f, Loader=yaml.FullLoader)

	yolo_img_size = cfg['yolo']['yolo_img_size']
	conf_thresh = cfg['yolo']['conf_thresh']
	iou_thresh = cfg['yolo']['iou_thresh']
	
	do_vis = cfg['function']['do_vis']
	do_print = cfg['function']['do_print']
	do_plot = cfg['function']['do_plot']
	do_selectROI = cfg['function']['do_selectROI']
	run_apriltag = cfg['function']['run_apriltag']
	run_csrt = cfg['function']['run_csrt']

	src_frame = cfg['tf']['src_frame']
	dst_frame = cfg['tf']['dst_frame']

	node_name = cfg['ros']['node_name']	
	rgb_topic = cfg['ros']['rgb_topic']	
	dep_topic = cfg['ros']['dep_topic']	
	info_topic = cfg['ros']['info_topic']	

	engine_file_path = "door_estimation/assets/yolov5m_{}/yolov5m_{}.engine". \
		format(yolo_img_size,yolo_img_size)
	plugin_library = "door_estimation/assets/yolov5m_{}/libmyplugins.so".format(yolo_img_size)


def show_marker(marker_array_, header_, pos_, ori_, scale_, color_, lifetime_, id_=0):
	marker_ = Marker()
	marker_.header = header_
	marker_.id = id_
	marker_.type = marker_.CUBE
	marker_.action = marker_.ADD

	marker_.pose.position.x = pos_[0]
	marker_.pose.position.y = pos_[1]
	marker_.pose.position.z = pos_[2]
	marker_.pose.orientation.x = ori_[1]
	marker_.pose.orientation.y = ori_[2]
	marker_.pose.orientation.z = ori_[3]
	marker_.pose.orientation.w = ori_[0]

	marker_.lifetime = rospy.Duration.from_sec(lifetime_)
	marker_.scale.x = scale_[0]
	marker_.scale.y = scale_[1]
	marker_.scale.z = scale_[2]
	marker_.color.a = 0.5
	red_, green_, blue_ = color_
	marker_.color.r = red_
	marker_.color.g = green_
	marker_.color.b = blue_
	marker_array_.markers.append(marker_)


class input_stack(object):
	"""
	This class wraps around the callback functions invoking door and 
	handle estimation model. It performs estimation using ROS 
	messages.
	"""
	def __init__(
		self, conf_thresh: float, 
		iou_thresh: float, 
		camera_info_msg: CameraInfo,
		do_vis: bool = False,
		do_plot: bool = False,
		do_selectROI: bool = False,
	):
		self.info = [0]*6						# img_height, img_width, fx, fy, cx, cy	
		
		self.conf = conf_thresh					# yolov5 inference confidence threshold
		self.iou = iou_thresh					# yolov5 IOU confidence threshold
		self.intrinsics = camera_info_msg.K		# camera intrinsics
		self.do_vis = do_vis
		self.do_print = do_print
		self.do_plot = do_plot
		self.do_selectROI = do_selectROI
		self.run_apriltag = run_apriltag
		self.run_csrt = run_csrt

		self.info[0] = camera_info_msg.height	# image height
		self.info[1] = camera_info_msg.width	# image width
		self.info[2] = self.intrinsics[0] 		# fx
		self.info[3] = self.intrinsics[4] 		# fy
		self.info[4] = self.intrinsics[2] 		# cx
		self.info[5] = self.intrinsics[5] 		# cy		
		
		self.history_stamp = 0
		self.listener = tf.TransformListener()

		# instantiate YoLov5TRT 
		ctypes.CDLL(plugin_library)
		self.yolov5_wrapper = YoLov5TRT(
			engine_file_path, self.conf, self.iou, self.info, self.do_print
		)
		# instantiate estimation pipeline
		self.estimator = DoorEstimation(
			self.intrinsics, self.info, self.yolov5_wrapper, 
			self.do_vis, self.do_print, self.do_selectROI, 
			self.run_apriltag, self.run_csrt, src_frame
		)
	
	def rgb_callback(self, rgb_msg: Image):
		self.rgb_msg = rgb_msg

	def dep_callback(self, dep_msg: Image):
		self.dep_msg = dep_msg

	def run_estimate(self, info_msg):
		"""
		description: 	Receive data from TfMessageFilter, process the data, 
						start door and handle estimation, and publish the
						estimation results.
		param:
			rgb_msg: 	Image, RGB image from ROS
			dep_msg: 	Image, depth image from ROS
			trans_quat: tuple[2], two lists contained, the translation vector 
						and the quaternion vector from the world frame to the
						depth camera frame
		"""
		if hasattr(self, 'rgb_msg') and hasattr(self, 'dep_msg'):
			if	self.history_stamp != self.rgb_msg.header.stamp.nsecs:
				nsecs = self.rgb_msg.header.stamp.nsecs
				self.history_stamp = nsecs
				ds = int(nsecs / 10**(len(str(nsecs))-1))
				# t_rgb = self.rgb_msg.header.stamp.secs + 0.1*ds
				# print('@@@@@@rgb_msg.header.stamp: ', self.rgb_msg.header.stamp.secs, '.', ds)

				# read tf message
				t = rospy.Time()
				trans_WC, quat_WC, t_tf = self.listener.lookupTransform(src_frame, dst_frame, t)
				#print('\n@@@@@@t_tf-t_rgb: ', t_tf-t_rgb)
				
				# build RGB arrray
				rgb_array = read_image_msg(self.rgb_msg, encoding="rgb8") 	# (480,640,3)

				# build depth arrray
				dep_array = read_image_msg(self.dep_msg, encoding="passthrough")
				dep_array = np.array(dep_array, dtype=np.float32)			# (480,640)

				# start door state estimation
				is_estimated = self.estimator.estimate(
					rgb_array, dep_array, trans_WC, quat_WC
				)
				
				# create results header
				result_header = Header()
				result_header.stamp = rospy.get_rostime()
				result_header.frame_id = src_frame

				# image messages
				rgb_val = self.estimator.rgb_val.astype("uint8")	# convert 32FC3 to rgb8
				rgb_scatter = self.estimator.rgb_scatter.astype("uint8")
				rgb_yolo = self.estimator.rgb_yolo.astype("uint8") 	
				rgb_csrt = self.estimator.rgb_csrt.astype("uint8") 	

				val_img_msg = build_image_msg(result_header, rgb_val, "rgb8")
				scatter_img_msg = build_image_msg(result_header, rgb_scatter, "rgb8")
				yolo_img_msg = build_image_msg(result_header, rgb_yolo, "rgb8")
				csrt_img_msg = build_image_msg(result_header, rgb_csrt, "rgb8")

				val_img_pub.publish(val_img_msg)
				scatter_img_pub.publish(scatter_img_msg)
				yolo_img_pub.publish(yolo_img_msg)
				csrt_img_pub.publish(csrt_img_msg)

				# poseArray message
				if self.estimator.count_h > 0:
					handle_C_array = PoseArray()
					handle_C_pose = Pose()
					# position (orientation is defaulted)
					handle_C_pose.position.x = self.estimator.handle_C_kf[0]
					handle_C_pose.position.y = self.estimator.handle_C_kf[1]
					handle_C_pose.position.z = self.estimator.handle_C_kf[2]

					handle_C_array.header.frame_id = dst_frame
					handle_C_array.header.stamp = rospy.get_rostime()
					handle_C_array.poses = [handle_C_pose]
					handle_C_pub.publish(handle_C_array)

				# publish estimation results
				if is_estimated:
					self.count_d = self.estimator.count_d
					if self.do_plot:
						# result messages
						door_est_result_msg = DoorEstimationResult()
						door_est_result_msg.header = result_header
						door_est_result_msg.elapsed_time = self.estimator.m_total
						door_est_result_msg.count = self.estimator.count_yolo
						# measurements
						if self.estimator.height_is_valid:
							door_est_result_msg.door_height = self.estimator.door_height
						if self.estimator.width_is_valid:
							door_est_result_msg.door_width = self.estimator.door_width
						# KF state vector
						door_est_result_msg.door_pos_W_kf.data = self.estimator.trans_WD_kf
						door_est_result_msg.normal_W_kf.data = self.estimator.normal_W_kf
						door_est_result_msg.door_height_kf = self.estimator.door_height_kf
						door_est_result_msg.door_width_kf = self.estimator.door_width_kf
						door_est_result_msg.handle_pos_D_kf.data = self.estimator.handle_D_kf
						door_est_result_msg.kf_measurement_noise = self.estimator.kf.R[0,0]
						# measurements
						if self.estimator.height_is_valid:
							door_est_result_msg.door_height = self.estimator.door_height
						else:
							door_est_result_msg.door_height = 0.
						if self.estimator.width_is_valid:		
							door_est_result_msg.door_width = self.estimator.door_width
						else:
							door_est_result_msg.door_width = 0.
						if self.estimator.handle_is_valid and self.estimator.width_is_valid:
							door_est_result_msg.handle_pos_D.data = self.estimator.handle_D
						else:
							door_est_result_msg.handle_pos_D.data = np.zeros(3)
						# ground truth from Apriltag
						door_est_result_msg.door_pos_W_gt_post.data = self.estimator.door_position_gt_W_post
						door_est_result_msg.door_nor_W_gt_post.data = self.estimator.door_normal_gt_W_post
						# results in C
						a,b,c = self.estimator.handle_C_kf
						door_est_result_msg.dis_CH = math.sqrt(a**2 + b**2 + c**2)
						# others
						if self.estimator.is_detected:
							yolo_det = np.asarray(self.estimator.det).reshape(-1)
							door_est_result_msg.yolo_det.data = yolo_det
						if self.count_d > 0 and len(self.estimator.xyxy_door_val) > 0:
							val_box = np.asarray(self.estimator.xyxy_door_val)	
							door_est_result_msg.val_box.data = val_box
						est_results_pub.publish(door_est_result_msg)

					if self.do_vis:	
						# pointcloud messages
						scene_points_xyz = self.estimator.scene_pcd_W
						scene_points_rgb = np.zeros((scene_points_xyz.shape[0], 3))
						scene_pointcloud_msg = build_pointcloud_msg(
							result_header, scene_points_xyz, scene_points_rgb
						)
						scene_pointcloud_pub.publish(scene_pointcloud_msg)

						# cube messages
						marker_array = MarkerArray()
						pos_door = self.estimator.trans_WD_kf
						ori = self.estimator.quat_WD_kf
						scale_door = [
							self.estimator.door_height_kf, self.estimator.door_width_kf, .08
						]
						scale_handle = (.1, .2, .08)

						lifetime = 1e4
						show_marker(
							marker_array, result_header, pos_door, ori, 
							scale_door, (0,1,0), lifetime, id_=0
						)
						if self.estimator.handle_D_kf[0] != 0.:
							pos_handle = self.estimator.handle_W_kf
							show_marker(
								marker_array, result_header, pos_handle, ori, 
								scale_handle, (1,0,0), lifetime, id_=1
							)
						marker_pub.publish(marker_array)

if __name__ == '__main__':
	rospy.init_node(node_name, disable_signals=True, anonymous=True)
	
	# publishers
	est_results_pub = rospy.Publisher('estimate/result', DoorEstimationResult, queue_size=1)
	yolo_img_pub = rospy.Publisher("estimate/image/yolo", Image, queue_size=1)
	val_img_pub = rospy.Publisher("estimate/image/validate", Image, queue_size=1)
	scatter_img_pub = rospy.Publisher("estimate/image/scatter", Image, queue_size=1)
	csrt_img_pub = rospy.Publisher("estimate/image/csrt", Image, queue_size=1)
	marker_pub = rospy.Publisher("estimate/cube/target", MarkerArray, queue_size=1)
	scene_pointcloud_pub = rospy.Publisher(
		"estimate/pointcloud/scene_pcd", PointCloud2, queue_size=1
	)
	handle_C_pub = rospy.Publisher("/tag_detections_pose", PoseArray, queue_size=1)

	# record the camera info
	rospy.loginfo('Please play the rosbag file')
	camera_info = rospy.wait_for_message(info_topic, CameraInfo)
	rospy.loginfo('Please pause the play')
	stack = input_stack(
		conf_thresh, iou_thresh, camera_info, do_vis, do_plot, do_selectROI
	)
	rospy.loginfo('Please continue the play')

	# subscribers
	rgb_sub = rospy.Subscriber(rgb_topic, Image, stack.rgb_callback)
	dep_sub = rospy.Subscriber(dep_topic, Image, stack.dep_callback)
	# run callback in a puppet subscriber
	info_sub = rospy.Subscriber(info_topic, CameraInfo, stack.run_estimate)

	# rospy.loginfo('Entering ros loop')
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print('Average time eplased is: {:0.3f}').format(stack.estimator.t_total)
		print("SHUTTING DOWN")
