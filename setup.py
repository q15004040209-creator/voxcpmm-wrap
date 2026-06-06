"""
voxcpmm-wrap
VoxCPM2 Python Wrapper - 无分词器多语言 TTS 封装
"""

from setuptools import setup, find_packages

setup(
    name="voxcpmm-wrap",
    version="1.0.0",
    description="VoxCPM2 Python Wrapper - Tokenizer-Free TTS for Multilingual Speech Generation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="q15004040209-creator",
    url="https://github.com/q15004040209-creator/voxcpmm-wrap",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "voxcpm",
        "soundfile>=0.12.1",
        "numpy>=1.24.0",
        "torch>=2.5.0",
    ],
    extras_require={
        "app": ["flask>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "voxcpmm-wrap=voxcpmm_wrap.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    license="Apache-2.0",
)