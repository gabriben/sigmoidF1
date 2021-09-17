from setuptools import setup

setup(name='sigmoidF1',
      version='0.1',
      description='sigmoidF1',
      url='https://github.com/gabriben/sigmoidF1',
      author='gabriben',
      author_email='gbndict@gmail.com',
      license='MIT',
      packages=['sigmoidF1'],
      install_requires=[
          'pandas',
          'sklearn',
          'mlflow',
          'numpy'
      ],
      zip_safe=False)
