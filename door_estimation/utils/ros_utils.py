# python
import numpy as np
import sys

# ros
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image


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