from setuptools import find_packages, setup

package_name = 'example_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haowen',
    maintainer_email='haowen@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [# 定义两个可执行节点
            'ball_detector_node = example_py.ball_detector_node:main',
            'ball_position_subscriber = example_py.ball_position_subscriber:main',
        ],
    },
)
