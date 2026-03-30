import torch
import torch.nn as nn
from einops import rearrange # 导入张量维度重排库

__all__ = ['LAE']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    Pad to 'same' shape outputs.
    自动计算填充大小，确保卷积输出尺寸与输入相同（"same"卷积）
    Args:
        k: 卷积核尺寸（整数或列表）
        p: 填充值（None表示自动计算）
        d: 膨胀率（默认1）
    Returns:
        填充值（整数或列表）
    """
    # 处理膨胀卷积：计算实际有效的卷积核尺寸
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    # 自动计算填充值（核尺寸的一半，向下取整）
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation).
    标准卷积模块：卷积 + 批归一化 + 激活函数
    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核尺寸（默认1）
        s: 卷积步长（默认1）
        p: 填充大小（None表示自动计算）
        g: 分组卷积组数（默认1）
        d: 膨胀率（默认1）
        act: 是否使用激活函数（True/False/自定义模块）
    """
    # 默认激活函数为SiLU（Swish）
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 创建卷积层
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False) # 不使用偏置（后续有BN）
        # 批归一化
        self.bn = nn.BatchNorm2d(c2)

        # 激活函数处理：
        # - act=True: 使用默认激活函数
        # - act=False: 使用恒等映射
        # - act=nn.Module: 使用自定义激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """标准前向传播：卷积 -> BN -> 激活函数"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合模式前向传播（跳过BN层，用于推理优化）"""
        return self.act(self.conv(x))


class LAE(nn.Module):
    """
    轻量自适应提取模块（Light-weight Adaptive Extraction）
    实现自适应下采样：通过注意力机制动态融合2×2区域

    Args:
        ch: 输入通道数
        group: 分组卷积的分组数（默认8）
    """
    def __init__(self, ch, group=8) -> None:
        super().__init__()

        # Softmax用于计算位置注意力权重（在最后一个维度归一化）
        self.softmax = nn.Softmax(dim=-1)

        # 注意力分支：捕获空间上下文信息
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),   # 3×3平均池化（保留局部上下文）
            Conv(ch, ch, k=1)   # 1×1卷积（整合通道信息）
        )

        # 下采样分支：3×3卷积实现2倍下采样
        # 输出通道扩展4倍（对应2×2区域的4个位置）
        # 使用分组卷积减少计算量（组数 = ch // group）
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        """
        前向传播过程：
        1. 生成位置注意力图
        2. 提取下采样特征
        3. 自适应融合特征
        """
        # 注意力分支：生成2×2区域的位置权重
        # 输入形状: [bs, ch, H, W] -> 输出形状: [bs, ch, H, W]
        # 重组张量：将空间维度分解为2×2的块
        # [bs, ch, 2*h, 2*w] -> [bs, ch, h, w, 4] (4对应2×2区域的4个位置)
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)

        # 在位置维度应用softmax：使4个位置的权重和为1
        att = self.softmax(att)

        # 下采样分支：提取特征
        # 输入形状: [bs, ch, H, W] -> 输出形状: [bs, 4*ch, H//2, W//2]
        # 重组张量：将通道维度分解为4组（对应2×2区域的4个位置）
        # [bs, (4 ch), h, w] -> [bs, ch, h, w, 4]
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)

        # 自适应融合：特征与注意力权重相乘，然后沿位置维度求和
        # [bs, ch, h, w, 4] * [bs, ch, h, w, 4] -> [bs, ch, h, w]
        x = torch.sum(x * att, dim=-1)

        # 输出形状: [bs, ch, H//2, W//2]
        return x


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = LAE(64)

    out = mobilenet_v1(image)
    print(out.size())