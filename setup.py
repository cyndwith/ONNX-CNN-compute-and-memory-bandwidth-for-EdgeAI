from setuptools import setup

setup(
    name='onnx-mac',
    version='1.0.0',
    description='ONNX tools to estimate compute and memory for CNN models',
    author='Dwith Chenna',
    author_email='cyndwith@ieee.org',
    url='https://github.com/cyndwith/ONNX-CNN-compute-and-memory-bandwidth-for-EdgeAI',
    packages=['onnx-mac'],
    install_requires=[
        # List your dependencies here
        onnx, 
        numpy,
        onnxruntime,
        plotly==4.14.3,
        kaleido
    ],
)
