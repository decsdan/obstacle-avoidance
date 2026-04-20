import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nav_server'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='scheiderd@carleton.edu',
    description='Custom Navigate action server and CLI client for the obstacle-avoidance fork',
    license='',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'navigation_server = nav_server.navigation_server:main',
            'navigate = nav_server.navigate_cli:main',
        ],
    },
)
