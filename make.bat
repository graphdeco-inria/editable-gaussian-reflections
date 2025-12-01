cd editable_gauss_refl/cuda

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%;%PATH%
set CUDAToolkit_ROOT=%CUDA_PATH%
set CMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe

cmake -S . -B build -G "Visual Studio 17 2022" ^
  -DCMAKE_CUDA_COMPILER="%CMAKE_CUDA_COMPILER%" ^
  -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%" ^
  -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release

cd ../..