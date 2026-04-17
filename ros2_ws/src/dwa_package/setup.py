import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dwa_package'

setup(
    name=package_name,
    version='0.0.0',
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
    description='Dynamic Window Approach local planner for TurtleBot4 obstacle avoidance',
    license='',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dwa_node = dwa_package.dwa_node:main',
        ],
    },
)
