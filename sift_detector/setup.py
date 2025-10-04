from setuptools import setup
import os
from glob import glob

package_name = 'sift_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Linghaowen',
    maintainer_email='your_email@example.com',
    description='SIFT-based template matching with RealSense for object detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sift_detector_node = sift_detector.sift_detector_node:main',
        ],
    },
)