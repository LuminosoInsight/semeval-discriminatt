from setuptools import setup

setup(
    name="semeval-discriminatt",
    version='0.1',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='rspeer@luminoso.com',
    platforms=["any"],
    description="Attempting SemEval-2018 task 10: Capturing Discriminative Attributes",
    packages=['discriminatt'],
    install_requires=['wordfreq', 'numpy', 'pandas', 'scikit-learn', 'attrs', 'conceptnet', 'nltk'],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
