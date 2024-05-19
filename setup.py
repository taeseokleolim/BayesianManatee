from setuptools import setup, find_packages

setup(
    name='bayesianmanatee',
    version='0.1',
    packages=find_packages(),
    description='Cython backed Bayesian Library without autodiff (do math for acceleration!)',
    author='Leo',
    author_email='taeseok.leo.lim@gmail.com',
    url='https://github.com/taeseokleolim/BayesianManatee',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
    ],
)