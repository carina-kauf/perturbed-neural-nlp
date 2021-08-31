#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()
	
requirements = [
	"numpy==1.19.5"
	"brain-score @ git+https://github.com/brain-score/brain-score.git",
	"result_caching @ git+https://github.com/brain-score/result_caching.git",
]


setup(
    name='perturbed-neural-nlp',
    version='0.1.0',
    description="How robust are language models to perturbations?",
    long_description=readme,
    author="Carina Kauf & Greta Tuckute",
    author_email='ckauf@mit.edu & gretatu@mit.edu',
    url='https://github.com/carina-kauf/perturbed-neural-nlp',
    packages=find_packages(),
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='computational neuroscience, human language, '
             'machine learning, deep neural networks',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6.2',
    ],
)