cd editable_gauss_refl/cuda
mkdir -p build
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
