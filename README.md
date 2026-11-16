# CPU MoE

本仓库分为三个部分：

| 目录 | 内容 |
| --- | --- |
| [cpu_moe/](cpu_moe/) | **CPU计算配套实现**：计算后端、Python运行时和依赖源码。使用CPU计算时下载完整目录，保持内部结构。 |
| [部署指南及参考脚本/](部署指南及参考脚本/) | 仅保留[参考启动脚本](部署指南及参考脚本/参考启动脚本.sh)、[脚本设定指南](部署指南及参考脚本/脚本设定指南.md)。 |
| [profile合成指南/](profile合成指南/) | CPU/NPU日志采集、时钟对齐、合成脚本和样例。需要分析profile时使用；[复制agent合成指令](profile合成指南/如何让agent合成profile.md)，默认只下载最终JSON到本地桌面。 |

参考启动脚本 [cpu_moe/普通CPU计算.sh](cpu_moe/普通CPU计算.sh) 默认使用单卡 BS1 配置，关闭 DSpark 和下一层专家预取。

## 下载什么

**准备CPU计算后端，下载完整的 [`cpu_moe/`](cpu_moe/) 目录即可。** 不要只取其中的 `kt-kernel/`，它还需要同级依赖与运行时文件。另从 [部署指南及参考脚本](部署指南及参考脚本/) 取使用说明、启动脚本；日志合成工具在独立的 [profile合成指南](profile合成指南/) 中。无需为了准备后端下载其他说明或日志样例。

## 两个仓库的关系

还必须单独准备 [vllm-ascend 的 `moe_cpu_compute_v1-on-0928` 分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)。**Ascend侧代码与CPU计算后端缺一不可**：前者负责CPU/NPU专家分工、LRC、搬运、预取和同步；本仓库 `cpu_moe/` 负责CPU专家计算。

### 共享 NZ 配对

使用配套的 vLLM 0.30.0、vLLM-Ascend 源码和本仓库源码。为目标架构编译 kt-kernel，确认原生扩展导出 `NZMOE`、派发 ABI 和线程池拓扑预检接口。

加载模型原有的 MXFP4 checkpoint 时，Ascend 生成 NZ 权重和 scale，CPU 后端与 NPU DMA 读取同一批缓冲区。配置 `cpu_moe.enabled=true` 会给所有符合条件的 MoE 层挂上 CPU 后端。计算线程与共享权重使用同一 NUMA 节点。构建和启动流程见 [脚本设定指南](部署指南及参考脚本/脚本设定指南.md)。

模型权重和编译后的运行库另行准备。更新 GitHub 时检查两边源码，有改动的一边同步更新，并记录新的配套版本。
