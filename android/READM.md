conda create -n android python=3.10
conda activate android 

export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.23.0.240601
export ANDROID_NDK_ROOT=/home/lei/android-ndk-r26d

export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export EXECUTORCH_ROOT=./executorch
export PYTHONPATH=$EXECUTORCH_ROOT/..

bash build.sh

python qnn.py -s 7500709c -m SM8650 -b $EXECUTORCH_ROOT/build-android
