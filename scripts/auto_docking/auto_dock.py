#!/usr/bin/env python

import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from neo_docking.srv import auto_docking
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Filter():
	# initialization
	def __init__(self):
		# node, server, publishers, subscribers
		self.node_name = 'auto_docking'
		rospy.init_node(self.node_name)
		frequency = 15
		self.rate = rospy.Rate(frequency)
		rospy.set_param('docking', False)
		server = rospy.Service(self.node_name, auto_docking, self.service_callback)
		self.mkr_pub = rospy.Publisher('ar_pose_filtered', PoseStamped, queue_size=1)
		self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
		self.pose_sub = rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.visual_callback)
		self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
		self.timer = rospy.Timer(rospy.Duration(0.01), self.low_pass_filter)
		# variables and interface for service
		self.STATION_NR = None
		self.vel = Twist()
		self.vel_vec = np.zeros(3)
		self.marker_pose = PoseStamped()
		self.marker_in_odom = PoseStamped()
		self.mkr_mat_corrected = None
		self.marker_list = []
		# data&params for sliding window
		self.window_size = 15
		self.base_to_odom = np.array([])
		self.translation = []			# of base_link in odom
		self.quaternion = []
		self.trans_marker_in_odom = []
		self.rot_marker_in_odom = []
		self.translation_window = []
		self.rotation_window = []
		# configurations
		# parameters for controllers
		self.node_name = 'auto_docking'
		self.kp_x = rospy.get_param('/'+self.node_name+'/kp/x')
		self.kp_y = rospy.get_param('/'+self.node_name+'/kp/y')
		self.kp_theta = rospy.get_param('/'+self.node_name+'/kp/theta')
		self.x_max = rospy.get_param('/'+self.node_name+'/velocity/linear/x/max')
		#self.x_min = rospy.get_param('/'+self.node_name+'/velocity/linear/x/min')
		self.y_max = rospy.get_param('/'+self.node_name+'/velocity/linear/y/max')
		#self.y_min = rospy.get_param('/'+self.node_name+'/velocity/linear/y/min')
		#self.tolerance = rospy.get_param('/'+self.node_name+'/tolerance/y')
		#self.angular_min = rospy.get_param('/'+self.node_name+'/velocity/angular/min')
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

	# solve quaternion from given matrix, knowing that 1st and 2nd angles are 0
	def quaternion_from_mat(self, mat, reference):
		# alpha = beta = 0
		# gamma = arctan(m[1][0]/m[0][0])
		euler_vec = self.euler_from_mat(mat, reference)
		q = quaternion_from_euler(euler_vec[0], euler_vec[1], euler_vec[2])
		return q

	# solve euler angle from given matrix, knowing that 1st and 2nd angles are 0
	def euler_from_mat(self, mat, reference):
		gamma = np.arctan(mat[1][0]/mat[0][0])
		if abs(gamma - reference) > (np.pi/2):
			gamma = -np.sign(gamma) * (np.pi - abs(gamma))
		return [0, 0, gamma]

	# callback of ar_track_alvar
	def visual_callback(self, ar_markers):
		self.marker_list = []
		for mkr in ar_markers.markers:
			# push every qualified & detected marker into the list
			if(not mkr.id in self.marker_list) and (mkr.id in self.defined_markers):
				self.marker_list.append(mkr.id)
			if(mkr.id == self.STATION_NR):
			# read pose data of the predefined marker
				self.marker_pose = mkr.pose
				self.marker_pose.header = mkr.header
				# do rotation, and remove unused information
				self.mkr_mat_corrected = self.do_correction(self.marker_pose)

	# transformation
	def do_correction(self, mkr_pose):
		mkr_quaternion = [mkr_pose.pose.orientation.x, mkr_pose.pose.orientation.y, mkr_pose.pose.orientation.z, mkr_pose.pose.orientation.w]
		mkr_euler = euler_from_quaternion(mkr_quaternion)
		mkr_rot_mat = self.mat_from_euler(mkr_euler)
		y_axis_of_cam = [[0],[1],[0]]
		y_axis_of_mkr = np.dot(mkr_rot_mat, y_axis_of_cam)
		# rotate the mkr coordinate system by 90 degrees
		correction_quaternion = np.zeros(4)
		correction_quaternion[0] = np.sin(0.785398)*y_axis_of_mkr[0]
		correction_quaternion[1] = np.sin(0.785398)*y_axis_of_mkr[1]
		correction_quaternion[2] = np.sin(0.785398)*y_axis_of_mkr[2]
		correction_quaternion[3] = np.cos(0.785398)
		correction_euler = euler_from_quaternion(correction_quaternion)
		correction_mat = self.mat_from_euler(correction_euler)
		mkr_rot_mat_corrected = np.dot(correction_mat, mkr_rot_mat)
		return mkr_rot_mat_corrected

	# sliding window
	# trans_or_rot = False for filtering translation
	# trans_or_rot = True for filtering rotation
	def sw_filter(self, data, window, size, trans_or_rot):
		window.append(data)
		if(len(window)>size):
			window.pop(0)
		# calculate the average
		if(len(window)<2):
			return window[0]
		ans = []
		l = len(window[0])
		for i in range(l):
			s = 0
			for vec in window:
				s += vec[i]
			ans.append(s/len(window))
		return ans

	# callback function for odometry
	def odom_callback(self, odom):		
		self.translation = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]
		self.quaternion = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
		euler = euler_from_quaternion(self.quaternion)
		mat = self.mat_from_euler(euler)
		self.base_to_odom = np.array(mat)
		if((odom.header.stamp - self.marker_pose.header.stamp) < rospy.Duration(0.08)):
			trans_marker_in_base = [-self.marker_pose.pose.position.x - 0.228, -self.marker_pose.pose.position.y + 0.015, self.marker_pose.pose.position.z + 0.461]
			self.trans_marker_in_odom = np.add(np.dot(self.base_to_odom, trans_marker_in_base), self.translation)
			self.rot_marker_in_odom = np.dot(self.base_to_odom, self.mkr_mat_corrected)
		if(self.marker_pose.header.stamp and len(self.trans_marker_in_odom)):		
			# wrapping marker_in_odom into rosmsg for visualization
			self.visualize(self.trans_marker_in_odom, self.rot_marker_in_odom)

	# check for limits of velocity
	def limit_check(self, vel, uper, lower):
		if(uper and abs(vel)>uper):
			vel = np.sign(vel) * uper
		elif(lower and abs(vel)<lower):
			vel = np.sign(vel) * lower
		return vel

	# resetting the orientation of every wheel
	def reset_wheels(self):
		x = Twist()
		null = Twist()
		x.linear.x = 0.01
		self.vel_pub.publish(x)
		time.sleep(0.2)
		self.vel_pub.publish(null)
		time.sleep(0.5)

	# callback function of timer, used for filtering velocities and sending them with 100Hz frequency
	def low_pass_filter(self, timer):
		self.vel.linear.x = 0.7*self.vel.linear.x + 0.3*self.vel_vec[0]
		self.vel.linear.y = 0.7*self.vel.linear.y + 0.3*self.vel_vec[1]
		self.vel.angular.z = 0.7*self.vel.angular.z + 0.3*self.vel_vec[2]
		if(len(self.translation_window)==self.window_size and rospy.get_param('docking')):
			self.vel_pub.publish(self.vel)
			#print(self.vel)

	# drive robot to marker
	def dock(self, trans_in_odom, rot_mat_in_odom):
		odom_to_base = self.base_to_odom.transpose()
		trans_in_base = np.dot(odom_to_base, np.subtract(trans_in_odom, self.translation))
		rot_mat_in_base = np.dot(odom_to_base, rot_mat_in_odom)
		rot_in_base = self.euler_from_mat(rot_mat_in_base, 0)
		if(self.window_size == 15):
			goal = 0.40
			tolerence = 0.005
		else:
			goal = 0.10
			tolerence = 0.001
		# controller
		vx = self.kp_x * (trans_in_base[0] + self.offset[0] + goal) * (abs(trans_in_base[0] + self.offset[0] + goal) > 0.01)
		vy = self.kp_y * trans_in_base[1] * (abs(trans_in_base[1] + self.offset[1]) > tolerence)
		vtheta = self.kp_theta * rot_in_base[2]	* (abs(np.degrees(rot_in_base[2] + self.offset[2])) > 50*tolerence)
		# restore velocities into member self.vel_vec
		self.vel_vec[0] = self.limit_check(vx, self.x_max, None)
		self.vel_vec[1] = self.limit_check(vy, self.y_max, None)
		self.vel_vec[2] = self.limit_check(vtheta, None, None)
		# TODO: choose another threshold for 1st part
		if (abs(self.vel_vec).sum() < 0.001):
			if(not self.window_size == 90):
				rospy.loginfo("Robot pose corrected.")
				self.window_size = 90
				# pause here if the reorganize of wheel orientation is still malfunctioning
				self.reset_wheels()
			else:
				self.window_size = 15
				self.translation_window = []
				self.rotation_window = []
				self.reset_wheels()
				rospy.loginfo("Process completed, error:")
				print(trans_in_base[0]+self.offset[0]+goal, trans_in_base[1], np.degrees(rot_in_base[2]))
				rospy.set_param('docking', False)

	# visualization of transformed marker pose
	def visualize(self, position_vec, rot_mat):
		q = self.quaternion_from_mat(rot_mat, euler_from_quaternion(self.quaternion)[2])
		mkr_pose_msg = PoseStamped()
		# give pose-msg the same stamp with mkr-msg
		mkr_pose_msg.header.stamp = self.marker_pose.header.stamp
		mkr_pose_msg.header.frame_id = 'odom'
		mkr_pose_msg.pose.position.x = position_vec[0]
		mkr_pose_msg.pose.position.y = position_vec[1]
		mkr_pose_msg.pose.position.z = position_vec[2]
		mkr_pose_msg.pose.orientation.x = q[0]
		mkr_pose_msg.pose.orientation.y = q[1]
		mkr_pose_msg.pose.orientation.z = q[2]
		mkr_pose_msg.pose.orientation.w = q[3]
		self.mkr_pub.publish(mkr_pose_msg)

	# callback function of service /auto_docking
	def service_callback(self, auto_docking):
		docking_state = rospy.get_param('docking')
		if(not docking_state):
			if(not auto_docking.station_nr in self.marker_list):
				return "Marker "+str(auto_docking.station_nr)+" not detected, please make sure robot is in a feasible area."
			else:
				self.STATION_NR = auto_docking.station_nr
				self.offset = [rospy.get_param('/'+self.node_name+'/model_'+str(self.STATION_NR)[0]+'/offset/x'), rospy.get_param('/'+self.node_name+'/model_'+str(self.STATION_NR)[0]+'/offset/y'), rospy.get_param('/'+self.node_name+'/model_'+str(self.STATION_NR)[0]+'/offset/theta')]
				rospy.set_param('docking', True)
				rospy.loginfo("Service request received.")
				return "Service requested."
		else:
			return "Robot is occupied now, request rejected."

if __name__ == '__main__':
	my_filter = Filter()
	while(not rospy.is_shutdown()):
		if(len(my_filter.rot_marker_in_odom) and rospy.get_param('docking')):
			# uncomment the following line if visualization of raw pose data is needed
			# my_filter.visualize([my_filter.marker_pose.pose.position.x, my_filter.marker_pose.pose.position.y, my_filter.marker_pose.pose.position.z], my_filter.mkr_mat_corrected)
			# sliding window for both translation and rotation
			translation = my_filter.trans_marker_in_odom
			rotation = my_filter.euler_from_mat(my_filter.rot_marker_in_odom, euler_from_quaternion(my_filter.quaternion)[2])
			translation_filtered = my_filter.sw_filter(translation, my_filter.translation_window, my_filter.window_size, False)
			rotation_filtered = my_filter.sw_filter(rotation, my_filter.rotation_window, my_filter.window_size, True)
			rot_mat_filtered = my_filter.mat_from_euler(rotation_filtered)
			my_filter.dock(translation_filtered, rot_mat_filtered)
		my_filter.rate.sleep()