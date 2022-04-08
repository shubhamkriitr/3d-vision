unzip ../eigen-3.4.0.zip
cd eigen-3.4.0/ && mkdir build && cd build && cmake ..
cd ../../
mkdir _build
cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DWITH_BENCHMARK=ON -DCMAKE_INSTALL_PREFIX=_install
cmake --build _build/ --target install -j 8
cmake --build _build/ --target pip-package
cmake --build _build/ --target install-pip-package