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
		self.bridge = CvBridge()

		image_topic = "/camera/depth/image_raw"
		pose_topic = '/amcl_pose'
		odom_topic = '/odom'
		goal_topic = '/move_base/goal'

		self.image_subscriber = rospy.Subscriber(image_topic, Image, self.on_frame_received)
		self.pose_subscriber = rospy.Subscriber(pose_topic, PoseWithCovarianceStamped, self.on_pose_received)
		self.odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.on_odom_received)
		self.goal_subscriber = rospy.Subscriber(goal_topic, PoseStamped, self.on_goal_received, queue_size=100)
		
		self.frame_idx = 1
		self.data_idx = 0
		self.rate = rate

		self.current_frame = None
		self.current_position = (0,0) # (x,y)
		self.current_orientation = 0 # (angle_x, angle_y, angle_z)
		self.current_velocity = (0,0,0) # (vx, vy, vz)
		self.current_goal = (0,0,0) # (x, y, angle_z)
		self.current_odom_position = (0,0) # (x,y)

		self.data_dir = 'data'
		for file_i in os.listdir(self.data_dir):
			if 'npz' in file_i:
				num = int(file_i.split(".")[0])
				self.data_idx = max(num, self.data_idx)
		self.data_idx += 1

	def on_frame_received(self, img_msg):
		self.current_frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")

		if self.frame_idx % self.rate == 0:
			file_name = str(self.data_idx) + '.npz'
			file_path = os.path.join(self.data_dir, file_name)
			np.savez(file_path, self.current_frame, np.array(self.current_velocity))
			self.data_idx += 1
		self.frame_idx += 1
		
		#cv2.imshow('Video', frame)
		#cv2.waitKey(1)

	def on_pose_received(self, pose):
		#rospy.loginfo('received pose {} '.format(pose))
		pose = pose.pose.pose
		self.current_position  = (pose.position.x, pose.position.y)
		self.current_orientation = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
		
		#print('Current AMCL position: ', self.current_position)
		#print('Current AMCL orientation: ' , self.current_orientation)	
	
	def on_odom_received(self, odom):
		odom_position = odom.pose.pose.position
		odom_orientation = odom.pose.pose.orientation
		linear_vel = odom.twist.twist.linear
		angular_vel = odom.twist.twist.angular
		self.current_velocity = (linear_vel.x, linear_vel.y, angular_vel.z)
		self.current_odom_position = (odom_position.x, odom_position.y)	
		self.current_odom_orientation = euler_from_quaternion([odom_orientation.x,
			odom_orientation.y, odom_orientation.z, odom_orientation.w])
		#print("Current odom position and orientation", self.current_odom_position,
		#		self.current_odom_orientation)
		# print('Current velocity: ', self.current_velocity)

	def on_goal_received(self, goal):
		goal_pose = goal.pose
		goal_position = (goal_pose.position.x, goal_pose.position.y)
		goal_orientation = euler_from_quaternion([goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z, goal_pose.orientation.w])
		self.current_goal = (goal_position[0], goal_position[1], goal_orientation[2])
		
		# print('Current goal: ', self.current_goal)

def main():
	# Initialize node
	rospy.init_node('trainer')
	rospy.loginfo('Recording training data')
	_ = Trainer(rate = 5)

	rospy.spin()


if __name__ == '__main__':
	main()
