# 参考profile：scale搬运修复版

本样例来自153服务器的 `dspark7_prefetch_scale_fix_20261109-231804`：单卡、BS1、CPU计算开启、DSpark7、下一层预取上限1个专家。上传的是搬运计数修正版，替换原17:41 pipeline参考样例。

这版实验的主要修改：

- 启动时将scale准备为NPU目标的内存布局和类型，推理时与专家权重一样直接按字节搬运，避开旧scale `Tensor.copy_` 路径中的布局处理及已观察到的内部GIL等待。当前采用额外预排副本；首次加载直接写最终布局的简化方案尚未实现。
- 合成时根据同次原始生命周期和DMA记录补正搬运数量。CPU标签如 `L1_E19_C19_T5`：E为不同CPU专家数，C为专家-token计算次数，T为本层按需搬运专家数。新增独立搬运计数轨道，`L1_T5` 表示本层搬5个，`L2_P1` 表示向下一层预取1个；一个专家的多次权重/scale拷贝不重复计数。

本次窗口包含本层搬运223批、336次专家传输，预取251批、251次专家传输，每批预取均为1个专家。所有原始NPU事件及CPU计算起止时间保持不变，仅补正计数并增加展示轨道。采样到的批内最大间隔为0.023微秒；跨批等待仍存在，不能解读为整条时间线没有空闲。

**本次提交仅更新参考profile和说明，运行代码暂未同步，不代表仓库当前代码已包含上述实验修改。**

[下载参考profile（gzip）](trace_view_cpu_compute.json.gz?raw=true)。JSON约263 MB，gzip约11.5 MB。解压后用MindStudio打开 `trace_view_cpu_compute.json`。

```bash
gzip -dk trace_view_cpu_compute.json.gz
sha256sum -c trace_view_cpu_compute.json.sha256
```

解压后的校验值见 [SHA256](trace_view_cpu_compute.json.sha256)。
