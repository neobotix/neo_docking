#!/usr/bin/env python

import sys
import time
import rospy
import tf2_ros
import numpy as np
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from neo_docking.srv import auto_docking
from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped, PoseStamped, Vector3, Twist

class Docking:
	# initialization
	def __init__(self):
		# initial values of parameters
		self.node_name = 'auto_docking'
		self.kp_x = rospy.get_param('/'+self.node_name+'/kp/x')
		self.kp_y = rospy.get_param('/'+self.node_name+'/kp/y')
		self.kp_theta = rospy.get_param('/'+self.node_name+'/kp/theta')
		self.x_max = rospy.get_param('/'+self.node_name+'/velocity/linear/x/max')
		self.x_min = rospy.get_param('/'+self.node_name+'/velocity/linear/x/min')
		self.y_max = rospy.get_param('/'+self.node_name+'/velocity/linear/y/max')
		self.y_min = rospy.get_param('/'+self.node_name+'/velocity/linear/y/min')
		self.tolerance = rospy.get_param('/'+self.node_name+'/tolerance/y')
		self.angular_min = rospy.get_param('/'+self.node_name+'/velocity/angular/min')
		self.start = 0
		self.base_pose = PoseStamped()
		self.base_marker_diff = PoseStamped()
		# initializing node, subscribers, publishers and servcer
		rospy.init_node(self.node_name)
		self.rate = rospy.Rate(15)
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
		listener = tf2_ros.TransformListener(self.tf_buffer)
		odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
		self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
		
		# in test
		self.LAST_STEP = False
		

	# functions for autonomous docking
	# callback function of odom_sub
	def odom_callback(self, odom):
		# restore measured odom info from nav_msgs/Odometry into geometry_msgs/PoseStamped format
		self.base_pose.header = odom.header
		self.base_pose.pose = odom.pose.pose

	# calculating displacement between marker and robot			
	def calculate_diff(self, filtered_pose):
		# remove the error due to displacement between map-frame and odom-frame
		map_to_base = self.tf_buffer.lookup_transform('base_link', 'map', rospy.Time(0), rospy.Duration(1.0))
		odom_map_diff = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(1.0))
		# compute difference between marker and base_link
		self.base_pose = tf2_geometry_msgs.do_transform_pose(self.base_pose, odom_map_diff)
		self.base_pose.header.frame_id = 'map'
		self.base_marker_diff.header.stamp = rospy.Time.now()
		self.base_marker_diff.header.frame_id = 'map'
		self.base_marker_diff.pose.position.x = filtered_pose.pose.position.x - self.base_pose.pose.position.x
		self.base_marker_diff.pose.position.y = filtered_pose.pose.position.y - self.base_pose.pose.position.y
		map_to_base.transform.translation = Vector3()
		self.base_marker_diff = tf2_geometry_msgs.do_transform_pose(self.base_marker_diff, map_to_base)
		filtered_pose_euler = euler_from_quaternion([filtered_pose.pose.orientation.x, filtered_pose.pose.orientation.y, filtered_pose.pose.orientation.z, filtered_pose.pose.orientation.w])
		base_pose_euler = euler_from_quaternion([self.base_pose.pose.orientation.x, self.base_pose.pose.orientation.y, self.base_pose.pose.orientation.z, self.base_pose.pose.orientation.w])
		# calculate the difference
		self.diff_x = self.base_marker_diff.pose.position.x
		# calibration of camera mounting		
		self.diff_y = self.base_marker_diff.pose.position.y
		self.diff_theta = filtered_pose_euler[2]-base_pose_euler[2]
		if(abs(self.diff_theta) > np.pi):
			self.diff_theta = self.diff_theta + np.sign(-self.diff_theta)*(2*np.pi)
		print("Difference: ["+str(self.diff_x)+", "+str(self.diff_y)+", "+str(np.degrees(self.diff_theta))+"]")
		
	# execute the first phase of docking process
	def locate(self):
		self.vel = Twist()
		# calculate the velocity needed for docking
		time_waited = time.time() - self.start
		# drive robot to where we start the visual servo process
		# visual servo would remove the error on x & y
		# in test
		self.vel.linear.x = 0
		self.vel.linear.y = 0
		# threshold in test, try to counter overshooting in first process
		if(abs(np.degrees(self.diff_theta)) < 2 or time_waited > 20):
			self.vel.angular.z = 0
			if(abs(self.diff_y) < 0.05 or time_waited > 10):
				self.vel.linear.y = 0
				if(abs(self.diff_x) < 0.722 or time_waited > 10):
					self.vel.linear.x = 0
				else:
					self.vel.linear.x = min(max(self.kp_x * self.diff_x, 2*self.x_min), self.x_max)
			else:
				self.vel.linear.y = self.kp_y * self.diff_y
				# defining the minimal cmd_vel on y-direction
				if abs(self.vel.linear.y) < 1.5 * self.y_min:
					self.vel.linear.y = 1.5 * self.y_min * np.sign(self.vel.linear.y)
				elif abs(self.vel.linear.y) > self.y_max:
					self.vel.linear.y = self.y_max * np.sign(self.vel.linear.y)
		# filter out shakes from AR tracking package
		elif(abs(np.degrees(self.diff_theta)) > 65):
			self.vel.angular.z = 0.005 * np.sign(self.diff_theta)
		else:
			self.vel.angular.z = self.kp_theta * self.diff_theta
			if(abs(self.vel.angular.z) < self.angular_min):
				self.vel.angular.z = self.angular_min * np.sign(self.vel.angular.z)
		state = self.vel.linear.x + self.vel.linear.y + self.vel.angular.z
		# check if the 1st phase of docking is done
		if(state == 0): 
			#print("start visual servo.")
			self.dock()
		else:
			self.vel_pub.publish(self.vel)

	# second phase of docking, serves for accuracy
	def dock(self):
		kp_x = rospy.get_param('/'+self.node_name+'/kp/x_fine')
		kp_y = rospy.get_param('/'+self.node_name+'/kp/y_fine')
		kp_theta = rospy.get_param('/'+self.node_name+'/kp/theta_fine')
		vel = Twist()
		# in case the 2nd docking process failed
		if(abs(np.degrees(self.diff_theta)) > 5 or self.diff_y > 0.02):
			self.start = time.time()
		# won't adjust vel.linear.x and vel.linear.y at the same time,
		# to avoid causing hardware damage
		
		# use a larger threshold when in last step, because the noise of visual feedback always makes vel.linear.y jumps between some value and 0
		# which destroyed the priority of y
		if(self.LAST_STEP):
			tolerance = self.tolerance
		else:
			tolerance = 0.5 * self.tolerance
		if(abs(self.diff_y) > tolerance):
			vel.linear.y = kp_y * self.diff_y
			if abs(vel.linear.y) < self.y_min:
				vel.linear.y = self.y_min * np.sign(vel.linear.y)
			elif abs(vel.linear.y > 0.8 * self.y_max):
				vel.linear.y = 0.8 * self.y_max * np.sign(vel.linear.y)
			vel.linear.x = 0
		else:
			self.LAST_STEP = True
			vel.linear.y = 0
			# correspondent: montage x = +25cm
			if(self.diff_x - 0.65 > 0.01):
				vel.linear.x = min(max(kp_x * (self.diff_x - 0.30), self.x_min), 2*self.x_min)
			else:
				vel.linear.x = 0
				vel.linear.y = 0
				if(abs(np.degrees(self.diff_theta)) < 0.05):
					vel.angular.z = 0
				else:
					vel.angular.z = 0.2 * self.kp_theta * self.diff_theta
					if(abs(vel.angular.z) < self.angular_min):
						vel.angular.z = self.angular_min * np.sign(vel.angular.z)
		self.vel_pub.publish(vel)
		# check if the process is done
		if(not (vel.linear.x + vel.linear.y + vel.angular.z)):
			rospy.set_param('docking', False)
			print("Connection established.")

class Filter():
# functions for pose transformation
	# initializations
	def __init__(self):
		self.node_name = 'auto_docking'
		self.marker_pose = PoseStamped()
		self.marker_pose_calibrated = PoseStamped()
		self.marker_list = []
		self.marker_list_printed = []
		self.STATION_NR = None
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
		listener = tf2_ros.TransformListener(self.tf_buffer)
		self.pose_sub = rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.marker_pose_calibration)
		self.filtered_pose_pub = rospy.Publisher('ar_pose_filtered', PoseStamped, queue_size=1)
		server = rospy.Service(self.node_name, auto_docking, self.service_callback)
		rospy.loginfo("auto_docking service is ready.")
		# load markers into a list, as reference of detection
		i = 0
		self.defined_markers = []
		while(True):
			i += 1
			try:
				marker = rospy.get_param('/ar_track_alvar/marker_'+str(i))
				marker = int(marker)
				self.defined_markers.append(marker)
			except:
				l = len(self.defined_markers)
				if(l):
					rospy.loginfo(str(l)+" marker(s) loaded.")
				else:
					rospy.loginfo("No marker is defined.")
				break

	# establish the rotation matrix from euler angle
	def mat_from_euler(self, euler):
		alpha = euler[0]
		beta = euler[1]
		gamma = euler[2]
		sa = np.sin(alpha)		# wrt x-axis
		ca = np.cos(alpha)
		sb = np.sin(beta)		# wrt y-axis
		cb = np.cos(beta)
		sr = np.sin(gamma)		# wrt z-axis
		cr = np.cos(gamma)
		# euler rotation matrix
		mat = [[cb*cr, sa*sb*cr - ca*sr, ca*sb*cr + sa*sr], [cb*sr, sa*sb*sr + ca*cr, ca*sb*sr - sa*cr], [-sb, sa*cb, ca*cb]]
		return mat		

	# transform measured marker pose into something comparable with robot coordinate system
	def do_calibration(self, marker):
		# correct the published orientation of marker
		# in convinience of calculating the error of orientation between marker & base_link
		cam_to_map = self.tf_buffer.lookup_transform('map', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
		marker_in_map = tf2_geometry_msgs.do_transform_pose(marker, cam_to_map)
		# do the rotation with euler rotation matrix
		marker_in_map_euler = euler_from_quaternion([marker_in_map.pose.orientation.x, marker_in_map.pose.orientation.y, marker_in_map.pose.orientation.z, marker_in_map.pose.orientation.w])
		marker_in_map_mat = self.mat_from_euler(marker_in_map_euler)
		y_axis_of_map = [[0], [1], [0]]
		axis_of_correction = np.dot(marker_in_map_mat, y_axis_of_map)
		correction_quaternion = np.zeros(4)
		correction_quaternion[0] = np.sin(0.785398)*axis_of_correction[0]
		correction_quaternion[1] = np.sin(0.785398)*axis_of_correction[1]
		correction_quaternion[2] = np.sin(0.785398)*axis_of_correction[2]
		correction_quaternion[3] = np.cos(0.785398)
		# calculate transformation with built-in function
		marker_correction = TransformStamped()
		marker_correction.header.stamp = rospy.Time.now()
		marker_correction.header.frame_id = 'map'
		marker_correction.transform.rotation.x = correction_quaternion[0]
		marker_correction.transform.rotation.y = correction_quaternion[1]
		marker_correction.transform.rotation.z = correction_quaternion[2]
		marker_correction.transform.rotation.w = correction_quaternion[3]
		marker_corrected = tf2_geometry_msgs.do_transform_pose(marker_in_map, marker_correction)
		marker_corrected.pose.position = marker_in_map.pose.position
		# ignore row & pitch of marker coordinate system
		euler_vec = euler_from_quaternion([marker_corrected.pose.orientation.x, marker_corrected.pose.orientation.y, marker_corrected.pose.orientation.z, marker_corrected.pose.orientation.w])
		orient_vec = quaternion_from_euler(0, 0, euler_vec[2])
		marker_corrected.pose.orientation.x = orient_vec[0]
		marker_corrected.pose.orientation.y = orient_vec[1]
		marker_corrected.pose.orientation.z = orient_vec[2]
		marker_corrected.pose.orientation.w = orient_vec[3]
		return marker_corrected

	# callback function: transforms measured marker pose into something comparable with robot coordinate system
	def marker_pose_calibration(self, ar_markers):
		#self.marker_pose_calibrated = PoseStamped()
		self.marker_list = []
		for mkr in ar_markers.markers:
			# push every qualified & detected marker into the list
			if(not mkr.id in self.marker_list) and (mkr.id in self.defined_markers):
				self.marker_list.append(mkr.id)
			if(mkr.id == self.STATION_NR):
			# read pose data of the predefined marker
				self.marker_pose = mkr.pose
				self.marker_pose.header.frame_id = 'camera_link'
				# do rotation, and remove unused information
				self.marker_pose_calibrated = self.do_calibration(self.marker_pose)

	# following functions serve for temporal Sliding Window
	# pack the PoseStamped into vector
	def vec_from_pose(self, pose):
		position_vec =  []
		orient_vec = []
		position_vec.append(pose.position.x)
		position_vec.append(pose.position.y)
		position_vec.append(pose.position.z)
		orient_vec.append(pose.orientation.x)
		orient_vec.append(pose.orientation.y)
		orient_vec.append(pose.orientation.z)
		orient_vec.append(pose.orientation.w)
		return [position_vec, orient_vec]

	# unpack the vector into PoseStamped
	def pose_from_vec(self, p_vec, o_vec):
		pose = PoseStamped()
		pose.header.stamp = rospy.Time.now()
		pose.header.frame_id = "map"
		pose.pose.position.x = p_vec[0]
		pose.pose.position.y = p_vec[1]
		pose.pose.position.z = p_vec[2]
		pose.pose.orientation.x = o_vec[0]
		pose.pose.orientation.y = o_vec[1]
		pose.pose.orientation.z = o_vec[2]
		pose.pose.orientation.w = o_vec[3]
		return pose

	# average value of each vec in a matrix
	def avr(self, mat):
		if(len(mat)<2):
			return mat[0]
		ans = []
		l = len(mat[0])
		for i in range(l):
			s = 0
			for vec in mat:
				s += vec[i]
			ans.append(s/len(mat))
		return ans

	# the callback function of service auto_docking
	def service_callback(self, auto_docking):
		try:
			docking_state = rospy.get_param('docking')
		except:
			rospy.set_param('docking', False)
			docking_state = False
		if(not docking_state):
			if(not auto_docking.station_nr in self.marker_list):
				return "Marker "+str(auto_docking.station_nr)+" not detected, please make sure robot is in a feasible area."
			else:
				self.STATION_NR = auto_docking.station_nr
				self.position_queue = []
				self.orientation_queue = []
				rospy.set_param('docking', True)
				print("Service request received.")
				return "Service requested."
		else:
			return "Robot is occupied now, request rejected."

if __name__ == '__main__':
	my_docking = Docking()
	my_filter = Filter()
	while(not rospy.is_shutdown()):
		# start docking when service is called
		# make sure marker is detected
		if(my_filter.marker_pose_calibrated.pose.position.x and rospy.get_param('docking')):
			# filtering the pose
			[position_vec, orient_vec] = my_filter.vec_from_pose(my_filter.marker_pose_calibrated.pose)
			euler_vec = euler_from_quaternion(orient_vec)
			my_filter.position_queue.append(position_vec)
			my_filter.orientation_queue.append(orient_vec)
			filtered_position_vec = my_filter.avr(my_filter.position_queue)
			filtered_orient_vec = my_filter.avr(my_filter.orientation_queue)
			filtered_pose = my_filter.pose_from_vec(filtered_position_vec, filtered_orient_vec)
			my_filter.marker_pose_calibrated = filtered_pose
			my_filter.filtered_pose_pub.publish(filtered_pose)
			if(len(my_filter.position_queue) == 15):
				my_filter.position_queue.pop(0)
				my_filter.orientation_queue.pop(0)
			# performing the docking procedure			
			my_docking.calculate_diff(filtered_pose)
			my_docking.locate()
			my_filter.marker_list_printed = []
		# if not docking, just print available markers(stations)
		elif(my_filter.marker_list and not sorted(my_filter.marker_list) == sorted(my_filter.marker_list_printed)):
			print("Marker(s) detected are:")
			print(sorted(my_filter.marker_list))
			my_filter.marker_list_printed = my_filter.marker_list
		my_docking.rate.sleep()
