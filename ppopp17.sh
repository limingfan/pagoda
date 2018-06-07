#!/bin/sh

echo $PWD

# Compile all benchmarks
cd matrixMul/pagoda_rand

echo "##  Compiling matrixMul pagoda  ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling matrixMul baseline  ##"
make clean
make

cd ..
cd cuda_fused_rand
echo "##  Compiling static fusion matrixMul baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread matrixMul baseline  ##"
make clean
make

cd ../..
cd convolution/pagoda_rand
echo "##  Compiling convolution pagoda  ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling convolution baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion convolution baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread convolution baseline  ##"
make clean
make


cd ../..
cd dct/pagoda_rand
echo "##  Compiling dct pagoda  ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling dct baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion dct baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread dct baseline  ##"
make clean
make

cd ../..
cd des/pagoda_rand
echo "##  Compiling des pagoda ## "
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling des baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion des baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread des baseline  ##"
make clean
make


cd ../..
cd filterbank/pagoda_rand
echo "## Compiling filterbank pagoda ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling filterbank baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion filterbank baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread filterbank baseline  ##"
make clean
make

cd ../..
cd beamformer/pagoda_rand
echo "## Compiling beamformer pagoda ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling beamformer baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion beamformer baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread beamformer baseline  ##"
make clean
make



cd ../..
cd mandelbrot/pagoda_rand
echo "##  Compiling mandelbrot pagoda  ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling mandelbrot baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion mandelbrot baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread mandelbrot baseline  ##"
make clean
make


cd ../..
cd multiwork/pagoda_rand
echo "## Compiling multiwork pagoda  ##"
make clean
make

cd ..
cd cuda_rand
echo "##  Compiling multiwork baseline  ##"
make clean
make

cd ..
cd cuda_fused
echo "##  Compiling static fusion multiwork baseline  ##"
make clean
make

cd ..
cd pthread
echo "##  Compiling pthread multiwork baseline  ##"
make clean
make

cd ../../
cd mandelbrot/pagoda_rand
# Running mandelbrot Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running mandelbrot CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running mandelbrot CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running mandelbrot pthread
sh run >> ../../test.txt
make clean

cd ../../
cd convolution/pagoda_rand
# Running convolution Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running convolution CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running convolution CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running convolution pthread
sh run >> ../../test.txt
make clean

cd ../../
cd dct/pagoda_rand
# Running dct Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running dct CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running dct CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running dct pthread
sh run >> ../../test.txt
make clean

cd ../../
cd filterbank/pagoda_rand
#Running filterbank Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running filterbank CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running filterbank CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running filterbank pthread
sh run >> ../../test.txt
make clean

cd ../../
cd beamformer/pagoda_rand
# Running beamformer Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running beamformer CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running beamformer CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running beamformer pthread
sh run >> ../../test.txt
make clean

cd ../../
cd matrixMul/pagoda_rand
# Running matrixMul Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running matrixMul CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused_rand
# Running matrixMul CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running matrixMul pthread
sh run >> ../../test.txt
make clean

cd ../../
cd des/pagoda_rand
# Running des Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running des CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running des CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running des pthread
sh run >> ../../test.txt
make clean


cd ../../
cd multiwork/pagoda_rand
# Running multiprogramming Pagoda
sh run >> ../../test.txt
make clean

cd ..
cd cuda_rand
# Running multiprogramming CUDA baseline
sh run >> ../../test.txt
make clean

cd ..
cd cuda_fused
# Running multiprogramming CUDA static fusion
sh run >> ../../test.txt
make clean

cd ..
cd pthread
# Running multiprogramming pthread
sh run >> ../../test.txt
make clean

cd ../../
python csv_generate.py

rm test.txt
echo "Completed"
