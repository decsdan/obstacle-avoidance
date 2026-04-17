from setuptools import find_packages, setup

package_name = 'd_star'

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
    maintainer='devin',
    maintainer_email='devin@todo.todo',
    description='D* Lite incremental pathfinding with dynamic replanning for TurtleBot4 navigation',
    license='',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'd_star_nav = d_star.d_star_nav:main',
            'live_visualizer = d_star.live_visualizer:main',
        ],
    },
)
