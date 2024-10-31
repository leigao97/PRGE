# Android Environment Setup with QNN SDK and ExecuTorch

### Step 1: Create Conda Environment

```bash
conda create -n android python=3.10
conda activate android
```

### Step 2: Install Qualcomm AI Engine Direct SDK and Android NDK
This code uses 2.23.0.240601: [QNN SDK website](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk?redirect=qdn)

This code uses r26d: [Android NDK page](https://developer.android.com/ndk)

### Step 3: Set Environment Variables

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.23.0.240601
export ANDROID_NDK_ROOT=./android-ndk-r26d

export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export EXECUTORCH_ROOT=./executorch
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

### Step 4: Build ExecuTorch

```bash
bash build.sh
```

### Step 5: Connect Android Device

```bash
adb get-serialno
```

### Step 6: Run Compile & Offload Script
```bash
python qnn.py -s 7500709c -m SM8650 -b $EXECUTORCH_ROOT/build-android
```

* -s: Device serial number
* -m: SoC identifier
* -b: Path to the Android build directory for ExecuTorch

### Check detailed tutorial from [ExecuTorch](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html)