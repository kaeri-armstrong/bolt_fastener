from setuptools import find_packages, setup
import glob

package_name = 'bolt_fastener'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob.glob(package_name+'/*.*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arm',
    maintainer_email='kwhb@kaeri.re.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bolt_fastener_node = bolt_fastener.bolt_fastener_node:main',
        ],
    },
)
