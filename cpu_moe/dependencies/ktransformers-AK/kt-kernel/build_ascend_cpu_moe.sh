#!/usr/bin/env bash
# Build in a fresh directory; never overwrite an active runtime package.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
BUILD=${1:?Usage: build_ascend_cpu_moe.sh /absolute/new/build/directory}
[[ "$BUILD" = /* ]] || { echo 'Build directory must be absolute' >&2; exit 2; }
[[ ! -e "$BUILD" ]] || { echo "Refusing existing build directory: $BUILD" >&2; exit 2; }
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh before building}"
command -v cmake >/dev/null
command -v pkg-config >/dev/null
pkg-config --exists hwloc
python -c 'import torch,torch_npu,pybind11,numpy,safetensors'
[[ -f "$SRC/../third_party/llama.cpp/CMakeLists.txt" && -f "$SRC/../third_party/pybind11/CMakeLists.txt" ]] || {
  echo 'Initialize third_party/llama.cpp and third_party/pybind11 first' >&2; exit 2;
}
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD/cmake" \
  -DCMAKE_BUILD_TYPE=Release -DKTRANSFORMERS_USE_ASCEND_NPU=ON \
  -DKTRANSFORMERS_USE_CUDA=OFF -DKTRANSFORMERS_USE_MUSA=OFF \
  -DKTRANSFORMERS_USE_ROCM=OFF -DLLAMA_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build "$BUILD/cmake" --target kt_kernel_ext -j"${CPUINFER_PARALLEL:-4}"
python - "$SRC" "$BUILD" <<'PY'
import pathlib,shutil,sys,sysconfig
src,build=map(pathlib.Path,sys.argv[1:])
package=build/'package/kt_kernel'
shutil.copytree(src/'python',package,ignore=shutil.ignore_patterns('*.so','__pycache__','*.pyc','*.pre-*'))
ext='kt_kernel_ext'+sysconfig.get_config_var('EXT_SUFFIX')
shutil.copy2(build/'cmake'/ext,package/ext)
PY
export PYTHONPATH="$BUILD/package:${PYTHONPATH:-}"
python -c 'import torch,torch_npu,kt_kernel; print(kt_kernel.__file__); print(kt_kernel.kt_kernel_ext.__file__); ext=kt_kernel.kt_kernel_ext; assert hasattr(ext.moe,"NZMOE"); assert all(hasattr(ext,name) for name in ("native_dispatch_api","native_cpu_handle","available_worker_pool_cpus")); assert hasattr(ext.WorkerPoolConfig(),"subpool_cpus")'
echo "KT_BUILD=$BUILD/package"
