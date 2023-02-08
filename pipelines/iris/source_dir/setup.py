from setuptools import setup, find_packages

setup(
    name='sagemaker-mlflow-iris',
    version='1.0',
    description='SageMaker MLFlow',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'mlflow==2.1.1',
    ],
)