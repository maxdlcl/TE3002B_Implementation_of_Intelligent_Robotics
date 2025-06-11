from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    node1 = Node(package='final_cocos',
                          executable='line_detection',
                            name="line_detection",
                          )
    
    node2 = Node(package='final_cocos',
                       executable='main_controller',
                       name="main_controller",
                       )
    
    node3 = Node(package='final_cocos',
                        executable='cmd_robot',
                        name="cmd_robot",
                        )
    
    l_d = LaunchDescription([node1, node2, node3])

    return l_d