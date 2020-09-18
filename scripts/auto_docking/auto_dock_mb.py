#!/usr/bin/env python

import time
import rospy
import tf2_ros
import actionlib
import numpy as np
import tf2_geometry_msgs

from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from neo_docking.srv import auto_docking
from neo_srvs.srv import ResetOmniWheels
from geometry_msgs.msg import PoseStamped, Twist
from ar_track_alvar_msgs.msg import AlvarMarkers
from actionlib_msgs.msg import GoalStatus as goal_status
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Filter():
	# initialization
	def __init__(self):
		# node, server
		self.node_name = 'auto_docking'
		rospy.init_node(self.node_name)
		frequency = 15
		self.rate = rospy.Rate(frequency)
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
		listener = tf2_ros.TransformListener(self.tf_buffer)
		self.odom_to_map = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(1.0))
		self.cam_to_base = self.tf_buffer.lookup_transform('base_link', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
		rospy.set_param('docking', False)
		rospy.set_param('undocking', False)
		server = rospy.Service(self.node_name, auto_docking, self.service_callback)
		server_undocking = rospy.Service('auto_undocking', auto_docking, self.service_undocking_callback)
		# variables and interface for service
		self.STATION_NR = None
		self.marker_pose = PoseStamped()
		self.marker_in_odom = PoseStamped()
		self.mkr_in_map_msg = PoseStamped()
		self.mkr_mat_corrected = None
		self.diff = rospy.get_param('auto_docking/differential_drive')
		self.p_gain = rospy.get_param('auto_docking/kp/x')
		self.max_vel_limit = rospy.get_param('auto_docking/max_vel_limit')
		# WARNING: PLEASE DO NOT CHANGE THE BELOW VALUE (docking_pose) WITHOUT THE CONSULTATION FROM NEOBOTIX. 
		self.undocking_pose = rospy.get_param('auto_docking/undocking_pose') # Safety is 50 cm for the robot to turn and do other actions
		self.marker_list = []
		# data&params for sliding window
		self.window_size = 30
		self.base_to_odom = np.array([])
		self.translation = []			# of base_link in odom
		self.quaternion = []
		self.translation_window = []
		self.quaternion_window = []
		# publishers, subscribers, action client
		self.mkr_pub = rospy.Publisher('ar_pose_filtered', PoseStamped, queue_size=1)
		self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
		self.pose_sub = rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.visual_callback)
		self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
		self.client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
		if(self.diff == False):
			if(self.client.wait_for_server()):
				rospy.loginfo("Action client server up.")
			rospy.wait_for_service('/kinematics_omnidrive/reset_omni_wheels')
			self.reset_wheels = rospy.ServiceProxy('/kinematics_omnidrive/reset_omni_wheels', ResetOmniWheels)
		# configuration
		# reading available markers
		self.defined_markers = []
		try:
			markers = rospy.get_param('/ar_track_alvar/markers')
			l = len(markers)
			for marker in markers:
				self.defined_markers.append(marker)
			if(l):
				rospy.loginfo(str(l)+" marker(s) loaded:"+str(self.defined_markers))
		except:
			rospy.loginfo("No marker is defined.")

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
	# [TODO]: take the rotation of self.cam_to_base, and put it into the calculation below, so the user can mount the camera the way they want
	#		  and update it in the urdf file, without affecting those transformations here.
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
	def sw_filter(self, data, window, size):
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

	# updating sliding window for the filter
	def sw_operator(self):
		trans_marker_in_map = [self.mkr_in_map_msg.pose.position.x, self.mkr_in_map_msg.pose.position.y, self.mkr_in_map_msg.pose.position.z]
		# rot_marker_in_map = euler_from_quaternion([self.mkr_in_map_msg.pose.orientation.x, self.mkr_in_map_msg.pose.orientation.y, self.mkr_in_map_msg.pose.orientation.z, self.mkr_in_map_msg.pose.orientation.w])
		quat_marker_in_map = [self.mkr_in_map_msg.pose.orientation.x, self.mkr_in_map_msg.pose.orientation.y, self.mkr_in_map_msg.pose.orientation.z, self.mkr_in_map_msg.pose.orientation.w]
		self.translation_filtered = self.sw_filter(trans_marker_in_map, self.translation_window, self.window_size)
		# self.rotation_filtered = self.sw_filter(rot_marker_in_map, self.rotation_window, self.window_size)
		quaternion_filtered = self.sw_filter(quat_marker_in_map, self.quaternion_window, self.window_size)
		self.rotation_filtered = euler_from_quaternion(quaternion_filtered)
		self.rot_mat_filtered = self.mat_from_euler(self.rotation_filtered)

	# callback function for odometry
	def odom_callback(self, odom):
		# need odom_to_map and base_in_odom to transfer marker_pose into map_frame.
		trans_marker_in_odom = []
		self.translation = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]
		self.quaternion = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
		self.odom_to_map = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(1.0))
		base_in_odom = PoseStamped()
		base_in_odom.header.frame_id = 'odom'
		base_in_odom.header.stamp = odom.header.stamp
		base_in_odom.pose = odom.pose.pose
		base_in_map = tf2_geometry_msgs.do_transform_pose(base_in_odom, self.odom_to_map)
		self.base_in_map_translation = [base_in_map.pose.position.x, base_in_map.pose.position.y, base_in_map.pose.position.z]		
		self.base_in_map_quaternion = [base_in_map.pose.orientation.x, base_in_map.pose.orientation.y, base_in_map.pose.orientation.z, base_in_map.pose.orientation.w]
		euler = euler_from_quaternion(self.quaternion)
		mat = self.mat_from_euler(euler)
		self.base_to_odom = np.array(mat)
		# dealing with time stamp issue.
		if((odom.header.stamp - self.marker_pose.header.stamp) < rospy.Duration(0.08)):
		# if(odom.header.stamp == self.marker_pose.header.stamp):
			trans_marker_in_base = [np.sign(self.cam_to_base.transform.translation.x) * self.marker_pose.pose.position.x + self.cam_to_base.transform.translation.x, -np.sign(self.cam_to_base.transform.translation.y) * self.marker_pose.pose.position.y + self.cam_to_base.transform.translation.y, self.marker_pose.pose.position.z + self.cam_to_base.transform.translation.z]
			trans_marker_in_odom = np.add(np.dot(self.base_to_odom, trans_marker_in_base), self.translation)
			rot_marker_in_odom = np.dot(self.base_to_odom, self.mkr_mat_corrected)
		if(len(self.marker_list) and len(trans_marker_in_odom)):		
			mkr_in_odom_msg = self.msg_wrapper(trans_marker_in_odom, rot_marker_in_odom, 'odom', self.marker_pose.header.stamp)
			self.mkr_in_map_msg = tf2_geometry_msgs.do_transform_pose(mkr_in_odom_msg, self.odom_to_map)
			# self.mkr_pub.publish(self.mkr_in_map_msg)

	# calculate the current goal according to visual feedback and pose of robot
	def calculate_goal(self, msg):
		if(self.window_size==30):
			offset = 0.30 + self.offset[0] + self.cam_to_base.transform.translation.x
		elif(self.window_size==45):
			offset = 0.15 + self.offset[0] + self.cam_to_base.transform.translation.x
		else:
			offset = self.offset[0]
		offset = -np.sign(self.cam_to_base.transform.translation.x) * offset
		offset_vec = [[offset], [0], [0]]
		rot_map_to_mkr = np.array(self.mat_from_euler(euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])))
		offset_in_map = np.dot(rot_map_to_mkr, offset_vec)
		offset_msg = PoseStamped()
		offset_msg.header.stamp = self.marker_pose.header.stamp
		offset_msg.header.frame_id = 'map'
		offset_msg.pose.position.x = offset_in_map[0][0] + msg.pose.position.x
		offset_msg.pose.position.y = offset_in_map[1][0] + msg.pose.position.y
		offset_msg.pose.position.z = offset_in_map[2][0] + msg.pose.position.z
		offset_msg.pose.orientation = msg.pose.orientation
		return offset_msg

	# drive robot to dock
	def dock(self, trans_in_map, rot_mat_in_map, stage):
		if(self.diff==False):
			rospy.loginfo("Performing docking for Omni-directional robot.")
		else:
			rospy.loginfo("Performing docking for differential-drive robot.")
		error = 0	
		filtered_pose_msg = self.msg_wrapper(trans_in_map, rot_mat_in_map, 'map', self.mkr_in_map_msg.header.stamp)
		offset = self.calculate_goal(filtered_pose_msg)
		if(stage == 1 or stage == 2 or (stage == 3 and self.diff == False)):
			goal = MoveBaseGoal()
			goal.target_pose.header.frame_id = 'map'
			goal.target_pose.pose = offset.pose
			goal.target_pose.pose.position.z = 0
			self.client.send_goal(goal)
			goal_reached = False
			return goal_reached
		else:
			cmd_vel = Twist()
			error = self.marker_pose.pose.position.x - self.offset[0]
			while error >= 0.01:
				error = self.marker_pose.pose.position.x - self.offset[0]
				if(self.p_gain*error > self.max_vel_limit):
					cmd_vel.linear.x = self.max_vel_limit
				else:
					cmd_vel.linear.x = self.p_gain*error
				self.vel_pub.publish(cmd_vel)
			cmd_vel.linear.x = 0
			self.vel_pub.publish(cmd_vel)
			goal_reached = True
			return goal_reached

	# wrapper of pose_msg
	def msg_wrapper(self, position_vec, rot_mat, frame_id, stamp):
		if(frame_id == 'map'):
			q = self.quaternion_from_mat(rot_mat, euler_from_quaternion(self.base_in_map_quaternion)[2])
		else:
			q = self.quaternion_from_mat(rot_mat, euler_from_quaternion(self.quaternion)[2])
		pose_msg = PoseStamped()
		# give pose-msg the same stamp with argument
		pose_msg.header.stamp = stamp
		pose_msg.header.frame_id = frame_id
		pose_msg.pose.position.x = position_vec[0]
		pose_msg.pose.position.y = position_vec[1]
		pose_msg.pose.position.z = position_vec[2]
		pose_msg.pose.orientation.x = q[0]
		pose_msg.pose.orientation.y = q[1]
		pose_msg.pose.orientation.z = q[2]
		pose_msg.pose.orientation.w = q[3]
		return pose_msg

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

		# callback function of service /auto_docking
	def service_undocking_callback(self, auto_docking):
		docking_state = rospy.get_param('undocking')
		if(not docking_state):
			if(not auto_docking.station_nr in self.marker_list):
				return "Marker "+str(auto_docking.station_nr)+" not detected, please make sure robot is in a feasible area."
			else:
				rospy.set_param('undocking', True)
				rospy.loginfo("Service request received. Please wait, until undocking is done")
				cmd_vel = Twist()
				error = self.undocking_pose - self.marker_pose.pose.position.x
			while error >= 0.01:
				error = self.undocking_pose - self.marker_pose.pose.position.x
				if(self.p_gain*-error<-self.max_vel_limit):
					cmd_vel.linear.x = -self.max_vel_limit
				else:
					cmd_vel.linear.x = self.p_gain*-error
				self.vel_pub.publish(cmd_vel)
			cmd_vel.linear.x = 0
			self.vel_pub.publish(cmd_vel)
			rospy.loginfo("Undocking is completed")	
			rospy.set_param('undocking', False)
		else:
			return "Robot is occupied now, request rejected."

if __name__ == '__main__':
	my_filter = Filter()
	stage = 1
	diff = rospy.get_param('auto_docking/differential_drive')
	while(not rospy.is_shutdown()):
		state = None
		if(my_filter.mkr_in_map_msg.header.stamp and rospy.get_param('docking')):
			# sliding window for both translation and rotation
			my_filter.sw_operator()
			# uncomment if visualization of filtered pose needed		
			filtered = my_filter.msg_wrapper(my_filter.translation_filtered, my_filter.rot_mat_filtered, 'map', my_filter.mkr_in_map_msg.header.stamp)
			my_filter.mkr_pub.publish(filtered)
			if (len(my_filter.translation_window) == my_filter.window_size):				
				goal_reached = my_filter.dock(my_filter.translation_filtered, my_filter.rot_mat_filtered, stage)
				rospy.loginfo("called dock.")
				if(state):
					state = None
				else:				
					state = my_filter.client.get_state()
				while(not rospy.is_shutdown() and not (state == goal_status.SUCCEEDED or goal_reached == True)):
					state = my_filter.client.get_state()
					my_filter.sw_operator()
					filtered = my_filter.msg_wrapper(my_filter.translation_filtered, my_filter.rot_mat_filtered, 'map', my_filter.mkr_in_map_msg.header.stamp)
					my_filter.mkr_pub.publish(filtered)
				# 3-stage docking process
				# if finished 1st stage, set window_size to 45 for 2nd stage.
				if(my_filter.window_size == 30):
					my_filter.window_size = 45
					stage = 2
					if(diff == False):
						my_filter.reset_wheels([0, 0, 0, 0])
				# if finished 2nd stage, set window_size to 50 for 3rd stage.
				elif(my_filter.window_size == 45):
					my_filter.window_size = 50
					stage = 3
					if(diff == False):
						my_filter.reset_wheels([0, 0, 0, 0])
				# if already 3rd stage, end the process and reset params
				else:
					my_filter.window_size = 30
					error_linear = np.subtract(my_filter.translation_filtered, my_filter.base_in_map_translation)
					error_angular = np.degrees(my_filter.rotation_filtered[2] - euler_from_quaternion(my_filter.base_in_map_quaternion)[2])
					rospy.loginfo("Connection established with error:")
					print(error_linear[0] - my_filter.offset[0],my_filter.marker_pose.pose.position.y - 0.0175, error_angular)
					rospy.set_param('diff_x', float(error_linear[0]))
					rospy.set_param('diff_y', float(my_filter.marker_pose.pose.position.y - 0.0175))
					rospy.set_param('diff_theta', float(error_angular))
					# Resetting the params
					rospy.set_param('docking', False)
					stage = 1
					goal_reached = False
				my_filter.translation_window = []
				my_filter.quaternion_window = []
		my_filter.rate.sleep()
