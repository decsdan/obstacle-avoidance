import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rl_local_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='scheiderd@carleton.edu',
    description='Reinforcement Learning based local planner for TurtleBot 4.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_local_planner_node = rl_local_planner.rl_local_planner_node:main',
        ],
    },
)
