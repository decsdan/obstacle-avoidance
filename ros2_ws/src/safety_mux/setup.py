import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'safety_mux'

setup(
    name=package_name,
    version='2.0.0',
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
    description='cmd_vel multiplexer and safety override. Latches a zero-velocity command on bumper, cliff, or sensor-staleness, independent of the orchestrator.',
    license='',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'safety_mux = safety_mux.safety_mux_node:main',
        ],
    },
)
