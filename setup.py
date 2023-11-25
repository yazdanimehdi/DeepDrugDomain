from setuptools import setup, find_packages
import os
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

# Read the contents of README file
with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open(os.path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Read the version from your package
package_dir = {'': 'deepdrugdomain'}
with open(os.path.join(package_dir[''], '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read())

setup(
    name='deepdrugdomain',
    version=__version__,
    author='Mehdi Yazdani-Jahromi',
    author_email='yazdani@ucf.edu',
    description='DeepDrugDomain: A versatile Python toolkit for streamlined preprocessing and accurate prediction of drug-target interactions and binding affinities, leveraging deep learning for advancing computational drug discovery.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yazdanimehdi/deepdrugdomain',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)
