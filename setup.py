from setuptools import setup, Extension

setup(
    name="simdlib",
    version="0.1.1",
    packages=["simdlib"],
    ext_modules=[
        Extension(
            name="simdlib_module",
            sources=["simdlib/simdlibmodule.c"],
            extra_compile_args=["-O3", "-arch", "arm64"],
        )
    ],
)
