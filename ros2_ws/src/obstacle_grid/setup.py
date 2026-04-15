from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obstacle_grid'

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
    maintainer='devin',
    maintainer_email='devin@todo.todo',
    description='Shared LIDAR-based obstacle grid with raycasting and temporal decay',
    license='',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'obstacle_grid = obstacle_grid.obstacle_grid_node:main',
        ],
    },
)
