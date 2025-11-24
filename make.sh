cd editable_gauss_refl/cuda
cmake -S . -B build -G Ninja
cmake --build build -- -j$(nproc)
