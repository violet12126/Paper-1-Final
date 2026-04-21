import torch
import torch.nn as nn
import numpy as np
import time
import os
from thop import profile, clever_format


# ================= 1.CNN 模型定义 =================
class CNN(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_res=True, kernel_size=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        current_channels = base_channels
        for _ in range(extra_conv_layers):
            self.features.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1))
            self.features.append(nn.BatchNorm2d(current_channels * 2))
            self.features.append(nn.LeakyReLU(0.1))
            self.features.append(nn.MaxPool2d(2, 2))
            current_channels *= 2

        if use_res:
            self.res = nn.Sequential(
                nn.Conv2d(current_channels, current_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(current_channels // 8, current_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.res = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if self.res is not None:
            res_mask = self.res(x)
            x = x * res_mask
        return self.classifier(x)


# ================= 2. 严谨的单张推理测速函数 =================
def measure_single_inference_time(model, device, num_warmup=50, num_iters=300):
    model.eval()

    # 构造一张单张图片的伪造输入 (Batch Size = 1, 3通道, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("开始预热 GPU (防止首次启动唤醒延迟)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    print(f"开始连续测量 {num_iters} 次单张图片推理时间...")
    times = []

    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == 'cuda':
                # CUDA 需要使用 Event 进行高精度异步测速
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()

                # 必须等待 GPU 把这行指令算完
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # 单位：毫秒(ms)
            else:
                # CPU 测速
                start_time = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - start_time) * 1000)

    avg_time = np.mean(times)
    return avg_time


# ================= 3. 主执行流程 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前测试设备: {device}\n")

    # 按照你训练时的参数初始化模型
    model = CNN(num_classes=6, base_channels=94, extra_conv_layers=0,
                use_res=False, dropout_rate=0.13046, kernel_size=5).to(device)

    # 1. 计算 FLOPs 和 Params (使用 thop 库)
    print("正在计算模型的 FLOPs 和 Params...")
    dummy_input_for_thop = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(model, inputs=(dummy_input_for_thop,), verbose=False)
    # 将长串数字格式化为友好的 M (Mega) 和 G (Giga) 格式
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    # 2. 执行测速获取 Latency
    avg_latency = measure_single_inference_time(model, device)

    # 3. 计算 FPS (Frames Per Second / 吞吐量)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    # 4. Accuracy (这里写入你之前测试出的真实准确率，方便统一输出)
    accuracy_str = "97.37%"

    # ================= 输出最终报告 =================
    print("\n" + "=" * 85)
    print(f"{'Method':<15} | {'Accuracy':<10} | {'Params':<12} | {'FLOPs':<12} | {'Latency (ms)':<15} | {'FPS':<10}")
    print("-" * 85)
    print(
        f"{'Ours (WSST-CNN)':<15} | {accuracy_str:<10} | {params_formatted:<12} | {macs_formatted:<12} | {avg_latency:<15.3f} | {fps:<10.1f}")
    print("=" * 85)

    print("\n[指标说明]")
    print("* Params: 模型参数量 (越小代表占用内存/磁盘空间越少)")
    print("* FLOPs: 浮点运算次数 (也就是 MACs，越小代表理论计算复杂度越低，是轻量化核心指标)")
    print("* Latency: 处理单张图片的端到端耗时 (毫秒，越小代表实时性越强)")
    print("* FPS: 每秒可处理的图片数量 (1000 / Latency，越大代表吞吐并发能力越强)")