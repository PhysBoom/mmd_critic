from setuptools import setup, find_packages

setup(
    name='mmd-critic',
    version='0.1.0',
    description='Python package for the MMD Critic method',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PhysBoom/mmd_critic',
    author='Matthew Chak',
    author_email='mchak@calpoly.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numexpr'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
