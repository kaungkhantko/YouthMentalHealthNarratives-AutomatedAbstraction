from setuptools import setup, find_packages

setup(
    name='youth-mental-health',
    version='0.1.0',
    description='Machine learning project for predicting youth mental health aspects.',
    author='Kaung Khant Ko',
    author_email='k.khantko@outlook.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'transformers',
        'tqdm',
        'huggingface_hub',
        'langchain',
        'numpy',
        'scikit-learn',
        'datasets',
        'sentencepiece',
        'accelerate',
        'tensorboard',
        'jupyter',
        'matplotlib',
        'seaborn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)