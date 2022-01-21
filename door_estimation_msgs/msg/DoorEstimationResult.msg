########## headers ############
# Standard header
std_msgs/Header header

# elapsed time 
float32 elapsed_time

# number of valid estimation
uint8 count



######## KF posterior ##########
# filtered door CoM position in World coordinate
std_msgs/Float64MultiArray door_pos_W_kf

# filtered Door normal in World
std_msgs/Float64MultiArray normal_W_kf

# filtered door height 
float32 door_height_kf

# filtered door width
float32 door_width_kf

# filtered door CoM position in World coordinate
std_msgs/Float64MultiArray handle_pos_D_kf



######## KF measurement ##########
# distance between the Camera and the Handle
float32 dis_CH

# measured door height 
float32 door_height

# measured door width
float32 door_width

# filtered door CoM position in World coordinate
std_msgs/Float64MultiArray handle_pos_D

# measurement noise
float32 kf_measurement_noise



######## bounding boxes ##########
# YOLO detection results (box, score and class) [n*6]
std_msgs/Float64MultiArray yolo_det

# validation 2D bounding box [1*4]
std_msgs/Float64MultiArray val_box



###### ground truth Apriltag #####
# door position ground truth in World
std_msgs/Float64MultiArray door_pos_W_gt_post
# door normal ground truth in World
std_msgs/Float64MultiArray door_nor_W_gt_post




## results in Camera coordinate
# measured door CoM position in Camera coordinate
# std_msgs/Float64MultiArray door_pos_C

## results in Camera coordinate
# measured door CoM position in Camera coordinate
# std_msgs/Float64MultiArray door_pos_C

# pointcloud of the whole scene
# sensor_msgs/PointCloud2 pcd_scene

# pointcloud of the door after RANSAC
# sensor_msgs/PointCloud2 pcd_door

# kalman gain in the kalman filter
# float32 kalman_gain