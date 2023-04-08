from setuptools import setup

setup(
    name='airi',
    version='1.0',
    description='tools for calculating RI retention indices',
    author='Lewis Geer',
    author_email='lewis.geer@nist.gov',
    packages=['airi'],
    install_requires=['requests', 'pandas', 'xlrd'],
    entry_points={
        'console_scripts': [
            'airi_calc = airi.airi_calc:main',
        ]
    }
)
