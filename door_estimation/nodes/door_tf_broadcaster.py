#!/usr/bin/env python  
# import roslib
# roslib.load_manifest('learning_tf')
import rospy
import tf
# import turtlesim.msg

def handle_turtle_pose(translation, rotation, doorname):
    br = tf.TransformBroadcaster()
    br.sendTransform(translation,
                     rotation,
                     rospy.Time.now(),
                     doorname,
                     '/depth_camera_front_camera')

if __name__ == '__main__':
    rospy.init_node('door_tf_broadcaster')
    doorname = 'estimated_door'
    translation = (1., 2., 3.)
    rotation = (0., 0., 0., 1.)
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        handle_turtle_pose(translation, rotation, doorname)

        # turtlename = rospy.get_param('~turtle')
        # rospy.Subscriber('/%s/pose' % turtlename,
        #                  turtlesim.msg.Pose,
        #                  handle_turtle_pose,
        #                  turtlename)
        # rospy.spin()
        rate.sleep()


# Problem:
# No transform from /world to /estimated_door