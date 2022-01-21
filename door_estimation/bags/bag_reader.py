import bagpy
from bagpy import bagreader
# import pandas as pd

b = bagreader('/media/rsl-admin/xavier_ssd/yif/bag/mayank_tf.bag')

# replace the topic name as per your need
LASER_MSG = b.message_by_topic('/tf')
LASER_MSG
# df_laser = pd.read_csv(LASER_MSG)
# df_laser # prints laser data in the form of pandas dataframe