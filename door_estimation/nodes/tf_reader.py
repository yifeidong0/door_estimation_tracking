import roslib
# roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv

if __name__ == '__main__':
    rospy.init_node('tf_listener', disable_signals=True)
    listener = tf.TransformListener()

    rate = rospy.Rate(1.0)
    t_start = rospy.get_time()
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/dynaarm_REALSENSE_depth_optical_frame', '/world', rospy.Time(0))
            # (trans2,rot2) = listener.lookupTransform('/estimated_door', '/dynaarm_REALSENSE_depth_optical_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        print('----1----', trans, rot)
        # print('----2----', trans2,rot2)

        if rospy.get_time() - t_start > 5:
            rospy.signal_shutdown("shutdown!")
            print("shutdown!")

        rate.sleep()