#!/usr/bin/env python
import rospy
from neo_docking.srv import auto_docking, auto_undocking
import time
import random
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus as goal_status
f = open('differential_docking_test.txt', 'w')


def main():
	rospy.init_node("Docking_iteration_test")
	rospy.loginfo("Let's start the test")
	rospy.wait_for_service ("auto_docking")
	rospy.wait_for_service ("auto_undocking")
	dock = rospy.ServiceProxy('auto_docking', auto_docking)
	un_dock = rospy.ServiceProxy('auto_undocking', auto_undocking)
	client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
	if(client.wait_for_server()):
				rospy.loginfo("Action client server up.")

	counter = 0
	stage  = 1
	move_basee = False
	state = None

	while(not rospy.is_shutdown() and counter < 3):
	# Do the dock 
		if(stage == 1):
			a = dock(10)
			time.sleep(0.5)
			move_base = False
			if(rospy.get_param('docking') == False):
				print >>f, rospy.get_param('diff_x'), rospy.get_param('diff_y'), rospy.get_param('diff_theta')
				stage = 2

	# Done ? Do the undock
		if(stage == 2):
			counter = counter+1
			b = un_dock(10)
			time.sleep(0.5)
			if(rospy.get_param('undocking') == False):
				stage = 3
		
	# Done ? Go to a random pose 
	# ToDo: Time-out functionality needs to be added so that the state machine does not get stuck the loop
		while(stage == 3):
			goal = MoveBaseGoal()
			goal.target_pose.header.frame_id = 'map'
			goal.target_pose.pose.position.x = round(random.uniform(-4.5, -5.0), 2) # Change the values based on your local frame! 
			goal.target_pose.pose.position.y = round(random.uniform(-3.0, -2.5), 2) # Change the values based on your local frame! 
			goal.target_pose.pose.orientation.x = 0
			goal.target_pose.pose.orientation.y = 0
			goal.target_pose.pose.orientation.z = 0.9999997
			goal.target_pose.pose.orientation.w = 0.0007963
			client.send_goal(goal)
			wait = client.wait_for_result()
			state = client.get_state()
			if(state == 3):
				stage = 1

	# Done ? Repeat the process	
		
if __name__ == '__main__':
	main()