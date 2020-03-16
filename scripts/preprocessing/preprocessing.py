#!/usr/bin/env python

import cv2
import rospy
import struct
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2

class Preprocess:

	def __init__(self):
		self.bridge = CvBridge()
		rospy.init_node('img_preprocessor')
		img_sub = rospy.Subscriber('camera1/depth/image_raw', Image, self.img_callback)
		self.img_pub = rospy.Publisher('camera1/depth/image_preprocessed', Image, queue_size=1)
		self.cv_image = []

	def img_callback(self, img):
		size = np.size(img.data)
		self.cv_image = self.bridge.imgmsg_to_cv2(img, 'bgr8')
		#rows, cols, channels
		self.shape = self.cv_image.shape

	def mask(self):
		# from the 213th we put the mask on raw img
		row = 213
		self.masked_image = self.cv_image
		self.masked_image[row:][:][:] = 0
		cv2.imshow("Image", self.masked_image)
		cv2.waitKey(3)

	def img_to_msg(self):
		ros_img_msg = self.bridge.cv2_to_imgmsg(self.masked_image)
		ros_img_msg.header.stamp = rospy.Time.now()
		ros_img_msg.header.frame_id = 'camera_link_optical'
		return ros_img_msg

if __name__ == '__main__':

	preprocess = Preprocess()
	rate = rospy.Rate(80)
	while(not rospy.is_shutdown()):
		if(np.size(preprocess.cv_image)):
			preprocess.mask()
			ros_img_msg = preprocess.img_to_msg()
			preprocess.img_pub.publish(ros_img_msg)
		rate.sleep()