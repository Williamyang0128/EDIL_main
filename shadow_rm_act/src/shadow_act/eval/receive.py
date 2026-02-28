import rospy
while(1):
	# 初始化节点
	rospy.init_node('parameter_demo', anonymous=True)

	# 获取列表参数
	step_action = rospy.get_param("/camera_publisher_left/step_action")
	step_numbers=step_action.split(",")
	list2= [int(num) for num in step_numbers]
	print(f"Retrieved list: {list2}")  # 输出: Retrieved list: [1, 2, 3, 4, 5]

# 保持节点运行
rospy.spin()
