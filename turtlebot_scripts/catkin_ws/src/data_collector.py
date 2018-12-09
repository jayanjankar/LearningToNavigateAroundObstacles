import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np 
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion , quaternion_from_euler 
	


class Trainer(object):
	def __init__(self, rate):
		 # Build data processor
		self.bridge = CvBridge()
		# Add publishers and subscribers
		image_topic = "/camera/depth/image_raw"
		pose_topic = '/amcl_pose'
		vel_topic = '/odom'
		goal_topic = '/move_base/goal'
		self.image_sub = rospy.Subscriber(image_topic, Image, self.on_frame_received)
		self.pose_subscriber = rospy.Subscriber(pose_topic, PoseWithCovarianceStamped, self.on_pose_received)
		self.vel_subscriber = rospy.Subscriber(vel_topic, Odometry, self.on_odom_received)
		self.goal_subscriber = rospy.Subscriber(goal_topic, PoseStamped, self.on_goal_received)
		
		self.idx = 0	
		self.rate = rate	
		self.current_position, self.current_orientation, self.current_velocity, self.current_goal = 0, 0, 0, 0

	def on_frame_received(self, img_msg):
		frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")
		# Display the resulting image
		#cv2.imshow('Video', frame)
		#cv2.waitKey(1)
		self.idx += 1
		if self.idx % self.rate == 0:
			np.save('data/' + str(self.idx) + '.npy' , frame)
			#cv2.imwrite('data/' + str(self.idx) + '.png' , frame)
			#print(frame)


    	def on_pose_received(self, pose):
        	#rospy.loginfo('received pose {} '.format(pose))
        	pose = pose.pose.pose
        	self.current_position  = (pose.position.x, pose.position.y)
		print('pos: ', self.current_position)
		self.current_orientation = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
		print('ori: ' , self.current_orientation)	
	
	def on_odom_received(self, odom):
		#rospy.loginfo('received odom {} '.format(odom))
		linear_vel = odom.twist.twist.linear
		angular_vel = odom.twist.twist.angular
		self.current_velocity = (linear_vel.x, linear_vel.y, angular_vel.z)
		print('vel: ', self.current_velocity)
		print('goal: ', self.current_goal)

	def on_goal_received(self, goal):
		print('goal')
		goal_pose = goal.pose
		goal_position = (goal_pose.position.x, goal_pose.positin.y)
		goal_orientation = euler_from_quaternion([goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z, goal_pose.orientation.w])
		self.current_goal = (goal_position[0], goal_position[1], goal_orientation[2])
		print('Goal: ', self.current_goal)

def main():
	# Initialize node
	rospy.init_node('trainer')
	rospy.loginfo('Recording training data')
	_ = Trainer(rate = 5)

	rospy.spin()


if __name__ == '__main__':
	main()
