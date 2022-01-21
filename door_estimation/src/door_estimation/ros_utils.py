import numpy as np
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField


def read_image_msg(input_msg: Image, encoding: str = "bgr8") -> np.ndarray:
    """Converts input ROS image message to numpy.

    Args:
        input_msg (Image): Input ROS Image message.
        encoding (str, optional): Encoding of the input message. Defaults to "bgr8".

    Returns:
        np.ndarray: The image matrix.
    """
    cv_bridge = CvBridge()
    try:
        cv_img = cv_bridge.imgmsg_to_cv2(input_msg, encoding)
        return cv_img
    except CvBridgeError as _cv_bridge_exception:
        rospy.logerr(_cv_bridge_exception)
        sys.exit(1)


def build_image_msg(header_msg: Header, img: np.ndarray, encoding: str = "bgr8") -> Image:
    """Converts numpy matrix to ROS image message.

    Args:
        header_msg (Header): The header message for the result.
        img (np.ndarray): Input image array.
        encoding (str, optional): Encoding of the input message. Defaults to "bgr8".

    Returns:
        Image: The converted ROS message.
    """
    cv_bridge = CvBridge()
    try:
        img_msg =  cv_bridge.cv2_to_imgmsg(img, encoding)
        img_msg.header = header_msg
        return img_msg
    except CvBridgeError as e:
        rospy.logerr(e)
        sys.exit(1)


def build_pointcloud_msg(header_msg: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    """Converts XYZ-RGB points to ROS sensor_msgs/PointCloud2 message.

    Args:
        header_msg (Header): The header message for the result.
        xyz (np.ndarray): XYZ points of shape (N, 3)
        rgb (np.ndarray): RGB points of shape (N, 3) with values 
            normalized (i.e. between 0 and 1).

    Returns:
        PointCloud2: The copnverted ROS message
    """
    # count number of points
    num_points = xyz.shape[0]
    xyzrgb = np.array(np.hstack([xyz, rgb]), dtype=np.float32)
    # create blank message
    pointcloud_msg = PointCloud2()
    pointcloud_msg.header = header_msg
    # add number of points
    pointcloud_msg.height = 1
    pointcloud_msg.width = num_points
    # add message fiels
    pointcloud_msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("r", 12, PointField.FLOAT32, 1),
        PointField("g", 16, PointField.FLOAT32, 1),
        PointField("b", 20, PointField.FLOAT32, 1),
    ]
    # add step information of buffer
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = 24
    pointcloud_msg.row_step = (
        pointcloud_msg.point_step * pointcloud_msg.width * pointcloud_msg.height
    )
    pointcloud_msg.is_dense = True
    # add buffer
    pointcloud_msg.data = xyzrgb.tostring()

    return pointcloud_msg