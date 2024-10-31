export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.23.0.240601
export ANDROID_NDK_ROOT=/home/lei/android-ndk-r26d
export EXECUTORCH_ROOT=./executorch

export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..

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

