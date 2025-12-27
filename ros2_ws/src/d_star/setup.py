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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'd_star_nav = d_star.d_star_nav:main',
            'odom = d_star.odom_subscriber:main',
            'visualizer = d_star.visualizer:main',
            'live_visualizer = d_star.live_visualizer:main',
        ],
    },
)
