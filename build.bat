@REM Linux
sudo apt install libeigen3-dev
pip install .

@REM Windows (with vcpkg)
vcpkg install eigen3:x64-windows
pip install . --config-settings cmake.define.CMAKE_TOOLCHAIN_FILE="C:/Users/shumi/codes/vcpkg/scripts/buildsystems/vcpkg.cmake"