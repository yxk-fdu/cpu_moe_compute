# CPU 计算后端

本目录保存 CPU MoE 后端及配套源码。与 [vllm-ascend 的 CPU分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) 配套：Ascend 侧负责专家分工、信号、搬运和同步；这里的 kt-kernel / CPUInfer 负责实际 CPU 专家计算。

| 路径 | 内容 |
| --- | --- |
| [dependencies/ktransformers-AK/kt-kernel/](dependencies/ktransformers-AK/kt-kernel) | C++ CPU专家算子、CPUInfer线程池、GGUF加载及构建入口。 |
| [kt_async_runtime/kt_kernel/](kt_async_runtime/kt_kernel) | Python运行时、后端适配与poll源码。 |
| [dependencies/ktransformers-AK/tools/](dependencies/ktransformers-AK/tools) | MXFP4逐层GGUF转换及校验工具。 |
| [dependencies/ktransformers-AK/third_party/](dependencies/ktransformers-AK/third_party) | 后端构建所用第三方源码和许可证。 |
| `merge_cpu_compute_latest.sh` / `merge_cpu_compute_trace.py` | CPU/NPU日志合成参考实现。当前使用入口见独立的profile合成指南。 |

## 共享 NZ 执行路径

选择 `weight_format="nz_mxfp4"` 时，Ascend 将最终 NZ 权重和 scale 的 Tensor 所有者交给本目录的 `NZMoEWrapper`。C++ `NZMOE` 和 Ascend 原生 DMA 使用同一批地址。CPU 后端按专家分组并复用重复选中专家的解包结果。此路径使用模型原有权重，按 [共享 NZ 配对说明](../README.md#共享-nz-配对) 编译和配置。

## 历史 GGUF 的 ARM 与 x86 执行路径

以下说明针对 `weight_format="gguf"` 的 MXFP4 路由专家。两种架构读取的权重均为 MXFP4；区别在于输入激活的量化方式和实际点积内核。GGUF 是文件容器，不能据此推断整个模型都是同一种精度。

| 项目 | ARM / AArch64 | x86 / x86_64 |
| --- | --- | --- |
| 匹配策略 | 后端 `kt_match_npu::enabled()` 固定返回 false，忽略匹配环境变量 | 由 `KT_MOE_MATCH_NPU_MXFP8` 是否存在决定 |
| 激活处理 | BF16 输入转换为 Q8_0；不执行额外的 MXFP8 匹配舍入 | 开启匹配时按 MXFP8 规则量化再反量化，保存为 F32；关闭时使用常规 Q8_0 路径 |
| GGUF 矩阵计算入口 | `kt_llamafile_sgemm()` → `kt_mxfp4_reuse::q8_matmul()` → ggml 的 `ggml_vec_dot_mxfp4_q8_0()` | 开启匹配时进入 `kt_mxfp4_reuse::match_matmul()` → `dot_gguf_mxfp4_mxfp8()` / 多行版本 |
| 硬件计算 | ARM NEON 整数点积、scale 还原及浮点累加 | 匹配内核在编译启用 AVX512F + AVX512BF16 时，将权重和激活转换为 BF16，执行 AVX512 BF16 点积；否则使用通用循环 |
| 开关设置 | 即使设置为 `1` 或 `0`，也不会进入匹配分支 | 设置为 `1` 或 `0` 都会开启；关闭必须 unset。结果在进程中首次调用时缓存 |

调用关系：

```text
GGUF MXFP4 专家权重
  ├─ ARM：Q8_0 激活 → llama.cpp/ggml NEON 内核
  └─ x86
       ├─ 匹配变量存在：MXFP8 规则处理激活 → AVX512 BF16 或通用匹配内核
       └─ 匹配变量不存在：Q8_0 激活 → 常规 GGUF 内核
```

为什么 x86 要额外走 MXFP8 匹配？优势是减少同一专家在 CPU 与 NPU 之间迁移时，由激活量化规则不同引起的输出变化：CPU 模仿 NPU 的 MXFP8 舍入规则，再利用已有的 AVX512 BF16 点积实现计算。因此它是在数值一致性与额外计算成本之间做取舍，不是让权重进一步压缩，也不保证比常规 Q8_0 路径更快；只有编译启用相应指令并运行在支持该指令的 x86 CPU 上，才能使用这条 SIMD 优化实现。

为什么 ARM 当前不走同一路径？原有 ARM 匹配分支落入通用逐元素循环，实测明显慢于已有的 llama.cpp/NEON 路径；实验性的匹配优化也未达到常规路径的耗时。因此当前 ARM 优先使用成熟的 Q8_0/NEON 实现恢复计算性能，把更高效的 MXFP8 匹配留作后续优化。这是当前实现的取舍，不代表 ARM 架构不能计算 MXFP8。

Q8_0 和 MXFP8 都是 8-bit 表示，但量化规则不同；ARM 采用 Q8_0 后，不承诺与 NPU MXFP8 输出逐位一致，整体模型精度仍需验证。这次调整的是 ARM 的后端派发策略，没有修改 x86 的量化、点积或多 token 复用实现。

这里没有使用 CPU 的原生 FP4×FP8 指令：压缩权重在计算时解包，再交给各架构的优化内核。`native_mxfp4` 是读取原始 FP4 数据和独立 scale 的另一条已有实现，不会因为这个修改自动转换为 GGUF；要使用这里的 ARM llama.cpp 路径，应显式配置 `weight_format="gguf"` 和对应 GGUF 路径。

### 构建与验证

修改的是 C++ 头文件，必须针对目标架构重新编译 `kt_kernel_ext`，再将新库安装到启动脚本的 `KT_BUILD` 所指运行包中；只同步源码或改环境变量不会更新旧 `.so`。已有 x86 二进制无需因 ARM 策略修改而重新部署。不要跨 ARM/x86 复用编译产物。

2026-11-09 在 ARM 118 验证：实际 CPUInfer / LlamafileMoEWrapper 后端，加载第 0 层真实 GGUF 专家权重，30 个线程、NUMA 3、1 专家、1 token；预热 10 次后测量 50 次，CPU 完整前向中位耗时约 85 μs。匹配变量 unset 与设置为 1 时输出 SHA256 一致，且最终权重缓冲区锁页成功。该结果包含 CPU 后端前向及提交/等待开销，不包含整模型、NPU 传输或端到端服务耗时；尚未以此修改完成整模型精度验证。

架构分支回归检查可在目标机器上编译并分别运行：

```bash
cd cpu_moe/dependencies/ktransformers-AK/kt-kernel
c++ -std=c++17 -O2 tests/test_mxfp8_arch_policy.cpp -o /tmp/test_mxfp8_arch_policy
env -u KT_MOE_MATCH_NPU_MXFP8 /tmp/test_mxfp8_arch_policy
env KT_MOE_MATCH_NPU_MXFP8=0 /tmp/test_mxfp8_arch_policy
env KT_MOE_MATCH_NPU_MXFP8=1 /tmp/test_mxfp8_arch_policy
```

## 普通CPU计算启动脚本

[普通CPU计算.sh](普通CPU计算.sh) 取自153之前跑通的无DSpark版本：单卡、BS1、CPU计算开启、下一层专家预取关闭。默认物理卡4、端口8144，计算线程与GGUF在NUMA1，30个计算线程。服务器同内容脚本为 `cpu_moe/run_cpu_basic.sh`。

脚本不传DSpark推测配置，`expert_prefetch_enabled=false`。`--safetensors-load-strategy prefetch` 仅为启动时checkpoint文件预读，不是下一层专家预取。迁移时通过 `WORKSPACE`、`ASCEND_REPO`、`KT_BUILD`、`MODEL_PATH`、`GGUF_PATH`、`LOG_ROOT` 指定路径；`NPU`和`PORT`可覆盖默认值。安装与其他关键参数见 [脚本设定指南](../部署指南及参考脚本/脚本设定指南.md)。

## 与153已验证 GGUF 环境的对应关系

153已完成整模型启动、推理和性能测试；CPU运行包为 `kt-native-profile-package`，计算采用GGUF，参考成本 **93.16 μs / expert-token（约90 μs）**。

本目录kt-kernel和Python/C运行时源码已从153实测工作区直接同步，包含原生派发接口、动态profile开关、MXFP4复用及MXFP8匹配相关实现。153 同步基线的架构策略已按上文增加 ARM GGUF 派发规则；当前纳入范围的文件校验值保存在 [version.json](version.json)，运行库、权重、日志、测试/benchmark及依赖自带Markdown不在该校验范围，第三方依赖源码也已对照153补齐并纳入校验。

固定配对：本仓库标签 [`cpu-moe-153-20261109-r2`](https://github.com/yxk-fdu/cpu_moe_compute/tree/cpu-moe-153-20261109-r2/cpu_moe) ↔ Ascend提交 [`7cf63ca`](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)，vLLM `0.30.0+empty`。**两边都要准备，缺一不可。** 本版本已集成原生派发接口，无需额外接口补丁。

`.so`、GGUF和NPU模型权重不在本目录中。历史脚本保留作参考，实际启动使用 [脚本设定指南](../部署指南及参考脚本/脚本设定指南.md)，日志与权重准备分别见 [profile合成指南](../profile合成指南/README.md) 和 [权重准备指南](../部署指南及参考脚本/权重准备指南.md)。
