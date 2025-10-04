from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取参数文件路径
    config = os.path.join(
        get_package_share_directory('your_package_name'),
        'config',
        'template_detector_params.yaml'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=config,
            description='Path to the parameters file'
        ),
        
        Node(
            package='your_package_name',
            executable='template_detector_depth',
            name='template_detector_depth',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            emulate_tty=True
        )
    ])