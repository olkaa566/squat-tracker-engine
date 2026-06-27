from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cpp_engine", 
        sources=["engine/cpp/src/matrix.cpp", "engine/cpp/src/bindings.cpp"],
        include_dirs=["engine/cpp/include", pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2'] 
    ),
]

setup(
    name="cpp_engine",
    ext_modules=ext_modules,
)