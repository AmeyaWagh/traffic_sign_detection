from setuptools import setup, find_packages

setup(
    name='traffic_light_detection',
    description="traffic sign detection package.",
    version='0.0.0',
    packages=find_packages(include=['traffic_light_detection', 'traffic_light_detection.*']),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)