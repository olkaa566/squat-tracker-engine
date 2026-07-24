from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_engine", 
        sources=[
            "engine/cpp/src/matrix.cpp", 
            "engine/cpp/src/multilayer_perceptron.cpp",
            "engine/cpp/src/geometry.cpp",
            "engine/cpp/src/biomechanics.cpp",
            "engine/cpp/src/bindings.cpp"
        ],
        include_dirs=["engine/cpp/include"],
        cxx_std=17, 
    ),
]

setup(
    name="cpp_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)