import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'scenario_runner'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'scenarios'), glob('scenarios/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='scheiderd@carleton.edu',
    description=('Episode lifecycle driver and dataset writer. Reads scenario '
                 'YAMLs, resets Gazebo, dispatches Navigate goals, detects '
                 'terminals, and writes matched rosbag2 + HDF5 sidecar '
                 'artifacts per episode.'),
    license='',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'scenario_runner = scenario_runner.scenario_runner_node:main',
        ],
    },
)
