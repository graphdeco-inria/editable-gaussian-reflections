cd gaussian_tracing/cuda
cmake -S . -B build -G Ninja
cmake --build build -- -j$(nproc)
