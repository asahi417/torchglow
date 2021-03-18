from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

VERSION = '0.0.0'
setup(
    name='torchglow',
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    license='MIT',
    description='Pytorch implementation of Glow.',
    url='https://github.com/asahi417/torchglow',
    # download_url="https://github.com/asahi417/torchglow/archive/v{}.tar.gz".format(VERSION),
    keywords=['normalizing flow', 'glow', 'machine learning', 'computer vision'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch==1.7.1",
        "tqdm",
        "torchvision==0.8.2",
        "keras",
        "requests",
        "tfrecord",
        "tensorboard",
        "gdown",
        "gensim",
        "pandas",
        "transformers",
        'numpy'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'torchglow-train-image = torchglow_cl.model_training:main_image',
            'torchglow-train-bert = torchglow_cl.model_training:main_bert',
            'torchglow-train-fasttext = torchglow_cl.model_training:main_fasttext'
        ]
    }
)
