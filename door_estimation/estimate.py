import os
import message_filters
import rospy
import numpy as np
# import matplotlib.pyplot as plt
import cv2
# import sys
import ctypes
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from door_estimation_msgs.msg import DoorEstimationResult

from estimation.main_estimation import main_estimation
from msg.msg_filter import TfMessageFilter
from yolov5.tensorrt.yolov5_trt import YoLov5TRT

rgb_path ='./input/rgb/'
dep_path ='./input/dep/'
rgb_name = '{}rgb.jpg'.format(rgb_path)
dep_name = '{}dep.jpg'.format(dep_path)
engine_file_path = "yolov5/tensorrt/trt7/yolov5m_trt7.engine"
plugin_library = "yolov5/tensorrt/trt7/libmyplugins.so"

node_name = "door_estimator"
rgb_topic = '/dynaarm_REALSENSE/color/image_raw'
dep_topic = '/dynaarm_REALSENSE/aligned_depth_to_color/image_raw'
info_topic = '/dynaarm_REALSENSE/color/camera_info'
tf_target_frame = '/dynaarm_REALSENSE_depth_optical_frame'
tf_source_frame = '/world'

class input_stack:
	rgb_array = np.array([])
	bgr_array = np.array([])
	dep_array = np.array([])
	intrinsics = np.array([])

	items = [0]*6	# img_height, img_width, fx, fy, cx, cy	
	count = 0
	door_prev = np.array([])

	def params(self, intr):
	    self.items[2] = intr[0] # fx
	    self.items[3] = intr[4] # fy
	    self.items[4] = intr[2] # cx
	    self.items[5] = intr[5] # cy

	def cb_init(self, camera_info):
		"""
		description: Receive camera info from subscriber.
		param:
			camera_info: CameraInfo, a class containing camera intrinsics,
						image sizes, etc
		"""
		print('START INITIALIZATION!')
		self.intrinsics = camera_info.K
		self.items[0] = camera_info.height
		self.items[1] = camera_info.width
		self.params(self.intrinsics)
		# instantiate a YoLov5TRT 
		ctypes.CDLL(plugin_library)
		self.yolov5_wrapper = YoLov5TRT(engine_file_path)

	def cb_img_tf(self, rgb, dep, trans_rot):
		"""
        description: Receive data from TfMessageFilter, process the data, 
					start door and handle estimation, and publish the
					estimation results.
        param:
            rgb: Image, a class in sensor_msgs that contains RGB image
            dep: Image, a class that contains depth image
            trans_rot: tuple, two lists contained, the translation vector 
					and the quaternion vector from the world frame to the
					depth camera frame
        """
		# print('start cb_img_tf!')
		rs_trans = trans_rot[0]
		rs_rot = trans_rot[1]

		# process the rgb-d images and write to path
		rgb_image = np.frombuffer(rgb.data, dtype=np.uint8). \
							reshape(rgb.height, rgb.width, -1)
		dep_image = np.frombuffer(dep.data, dtype=np.uint16). \
							reshape(dep.height, dep.width, -1)

		self.rgb_array = np.array(rgb_image, dtype=np.float32) # size: h*w*3
		self.bgr_array = cv2.cvtColor(self.rgb_array, cv2.COLOR_RGB2BGR)
		self.dep_array = np.array(dep_image, dtype=np.float32) # size: h*w*1
		self.dep_array = self.dep_array.reshape((rgb.height, rgb.width)) # size: h*w
		
		cv2.imwrite(rgb_name, self.bgr_array)
		cv2.imwrite(dep_name, self.dep_array)

		# start door and handle estimation
		print('\nFRAME NO.: ', self.count)
		self.door_prev, quit, elapsed_time = main_estimation(self.rgb_array, self.dep_array, 
									self.items, rs_trans, rs_rot, 
									self.count, self.door_prev, self.yolov5_wrapper)

		# publish estimation results
		if not quit:
			self.count += 1
			door_est_result = DoorEstimationResult()
			door_est_result.elapsed_time = elapsed_time
			door_est_result.door_com_kf.data = self.door_prev
			door_est_result.count = self.count
			est_results_pub.publish(door_est_result)

if __name__ == '__main__':
	if len(os.listdir(dep_path)) + len(os.listdir(rgb_path)) > 0:
		os.system('rm input/dep/dep.jpg')
		os.system('rm input/rgb/rgb.jpg')

	stack = input_stack()
	rospy.init_node(node_name, disable_signals=True, 
						anonymous=True)

	# subscribers
	rgb_sub = message_filters.Subscriber(rgb_topic, Image)
	dep_sub = message_filters.Subscriber(dep_topic, Image)

	# publishers
	est_results_pub = rospy.Publisher('estimate/result', 
									DoorEstimationResult, queue_size=10)
	
	# record the camera info
	camera_info = rospy.wait_for_message(info_topic, CameraInfo)
	stack.cb_init(camera_info)

	# synchronize tf, rgb and depth images
	ts_img_tf = TfMessageFilter([rgb_sub, dep_sub], 
						tf_source_frame, tf_target_frame, queue_size=500)
	ts_img_tf.registerCallback(stack.cb_img_tf)

	rospy.loginfo('Entering ros loop')

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("SHUTTING DOWN")