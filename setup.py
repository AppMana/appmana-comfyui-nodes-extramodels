from setuptools import setup, find_packages
import os.path

setup(
    name="comfyui_extra_models",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    include_package_data=True,
    entry_points={
        'comfyui.custom_nodes': [
            'comfyui_extra_models = comfyui_extra_models',
        ],
    },
    package_data={
        'comfyui_extra_models': ['**/*.json', '**/spiece.model']
    },
)