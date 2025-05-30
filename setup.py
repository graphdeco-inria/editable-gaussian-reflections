import importlib.util
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = None
exec(open("gaussian_tracing/version.py", "r").read())


def get_package_path(package_name):
    """
    Returns the filesystem path to the root of the given package,
    whether it's installed in editable mode or normally.

    :param package_name: The name of the package to locate.
    :return: Absolute path to the package directory.
    :raises ImportError: If the package cannot be found.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot find package '{package_name}'")

    # If it's a module (not a package), return the file's directory
    if spec.submodule_search_locations is None:
        return os.path.dirname(spec.origin)

    # It's a package: return the top-level package path
    return os.path.abspath(spec.submodule_search_locations[0])


# Custom build extension to build the OptiX tracing kernel
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Run the original build_extensions
        super().build_extensions()

        # Build OptiX library
        pkg_source = os.path.dirname(os.path.abspath(__file__))
        pkg_target = get_package_path("gaussian_tracing")
        if not os.path.exists(pkg_target):
            os.makedirs(pkg_target, exist_ok=True)

        os.system(
            f"mkdir -p {pkg_source}/gaussian_tracing/cuda/build && cd {pkg_source}/gaussian_tracing/cuda/build && cmake .. && make"
        )
        os.system(
            f"cp {pkg_source}/gaussian_tracing/cuda/build/libgausstracer.so {pkg_target}"
        )
        os.system(
            f"cp {pkg_source}/gaussian_tracing/cuda/build/raytracer_config.py {pkg_target}"
        )


setup(
    name="gaussian_tracing",
    version=__version__,
    description=" Python package for differentiable tracing of gaussians",
    keywords="gaussian, raytracing, cuda",
    python_requires=">=3.10",
    install_requires=[
        "ninja",
        "numpy<2.0.0",
        "torch",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
        ],
    },
    ext_modules=[
        CUDAExtension(
            name="gaussian_tracing._C",
            sources=["gaussian_tracing/cuda/ext.cpp"],
            include_dirs=[],
        ),
    ],
    cmdclass={"build_ext": CustomBuildExtension},
    packages=find_packages(),
)
