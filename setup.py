from setuptools import setup, find_packages  # Always prefer setuptools over distutils

setup(
    name='interpolation',
    version='2.1.2',
    description='Interpolation in Python',
    url='https://github.com/econforge/interpolation.py',
    author='Chase Coleman, Spencer Lyon and Pablo Winant',
    author_email='pablo.winant@gmail.com',
    license='BSD-2',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=["numba", "numpy", "scipy", "tempita"],
    zip_safe=False
)
