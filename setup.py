import importlib.util
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = None
exec(open("editable_gauss_refl/version.py", "r").read())


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
        pkg_target = get_package_path("editable_gauss_refl")
        if not os.path.exists(pkg_target):
            os.makedirs(pkg_target, exist_ok=True)

        os.system(f"mkdir -p {pkg_source}/editable_gauss_refl/cuda/build && cd {pkg_source}/editable_gauss_refl/cuda/build && cmake .. && make")


setup(
    name="editable_gauss_refl",
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
            "clang-format",
            "pytest",
            "ruff",
        ],
    },
    ext_modules=[
        CUDAExtension(
            name="editable_gauss_refl._C",
            sources=["editable_gauss_refl/cuda/ext.cpp"],
            include_dirs=[],
        ),
    ],
    cmdclass={"build_ext": CustomBuildExtension},
    packages=find_packages(),
)
