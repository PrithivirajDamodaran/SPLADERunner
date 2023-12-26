from setuptools import setup, find_packages

setup(
    name='SPLADERunner', 
    version='0.0.4', 
    packages=find_packages(),
    install_requires=[
        'tokenizers',
        'onnxruntime',
        'numpy',
        'requests',
        'tqdm'
    ],  
    author='Prithivi Da',
    author_email='',
    description='Ultralight and Fast wrapper for the unofficial implementation of SPLADEv2 models for your search & retrieval pipelines. Models and Library created by Prithivi Da, For PRs and Collaboration to checkout the readme.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PrithivirajDamodaran/SPLADERunner',  
    license='CC-BY-SA-4.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
