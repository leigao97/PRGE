if [ -z "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT is not set."
    exit 1
fi

if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT is not set."
    exit 1
fi

if [ -z "$EXECUTORCH_ROOT" ]; then
    echo "Error: EXECUTORCH_ROOT is not set."
    exit 1
fi

if [ -z "$LD_LIBRARY_PATH" ]; then
    echo "Error: LD_LIBRARY_PATH is not set."
    exit 1
fi

if [ -z "$PYTHONPATH" ]; then
    echo "Error: PYTHONPATH is not set."
    exit 1
fi

git submodule sync
git submodule update --init --recursive

cp ./custom_kernels/op_randn.cpp $EXECUTORCH_ROOT/kernels/portable/cpu/
declaration="
- op: randn.out
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::randn_out
  tags: nondeterministic_seeded"

file="$EXECUTORCH_ROOT/kernels/portable/functions.yaml"
if ! grep -q "randn.out" "$file"; then
  # If not, append the declaration to the file
  echo "$declaration" >> "$file"
  echo "randn op declaration appended to the file."
else
  echo "randn op declaration already exists in the file."
fi

cd $EXECUTORCH_ROOT
./install_requirements.sh

# (rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)
# cmake --build cmake-out --target executor_runner -j9

./backends/qualcomm/scripts/build.sh 
cp schema/program.fbs exir/_serialize/program.fbs
cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs

