from setuptools import find_packages, setup

package_name = 'jps'

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
    maintainer='emekar',
    maintainer_email='emekar@todo.todo',
    description='Jump Point Search pathfinding algorithm for TurtleBot4 navigation',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'jps_nav = jps.jps_nav:main',
            'odom = jps.odom_subscriber:main',
            'jps_visualizer = jps.jps_visualizer:main',
        ],
    },
)