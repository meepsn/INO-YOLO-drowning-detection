"""
An implementation of GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
The train script of the model is similar to that of MobileNetV3
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch
"""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, SqueezeExcite    # 从timm库导入DropPath和SE模块
from timm.models.layers import CondConv2d, hard_sigmoid, DropPath

__all__ = ['GhostC3k2']

# 创建SE层，使用hard_sigmoid激活函数，减少计算量
_SE_LAYER = partial(SqueezeExcite, gate_fn=hard_sigmoid, divisor=4)


class DynamicConv(nn.Module):
    """
    动态卷积层（多专家条件卷积）
    创新点：根据输入特征动态组合多个卷积核
    - 解决静态卷积核对不同输入适应性不足的问题
    - 平衡模型容量和计算效率

    参数：
    num_experts: 专家卷积核数量（默认4个）
    routing: 路由网络生成专家权重
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()

        # 路由网络：将输入特征映射到专家权重空间
        self.routing = nn.Linear(in_features, num_experts)

        # 条件卷积：根据路由权重组合多个专家卷积核
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        # 全局平均池化获取特征向量
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        # 生成路由权重
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        # 应用动态卷积
        x = self.cond_conv(x, routing_weights)
        return x


class ConvBnAct(nn.Module):
    """
    增强型卷积块（卷积+BN+激活+可选残差）
    设计特点：
    - 使用动态卷积替代标准卷积
    - 支持DropPath正则化
    - 自动判断是否添加残差连接

    残差条件：stride=1 且 in_chs=out_chs
    """
    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='',
            skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0., num_experts=4):
        super(ConvBnAct, self).__init__()
        # 残差连接条件判断：输入输出通道数相同且步长为1
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        # 随机深度丢弃率
        self.drop_path_rate = drop_path_rate

        # 使用动态卷积替代标准卷积
        # self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.conv = DynamicConv(in_chs, out_chs, kernel_size, stride, dilation=dilation, padding=pad_type,
                                num_experts=num_experts)
        self.bn1 = norm_layer(out_chs)  # 批归一化
        self.act1 = act_layer() # 激活函数

    def feature_info(self, location):
        """
        特征提取点定义（用于特征金字塔网络）
        expansion: 卷积激活后的特征（高分辨率细节）
        bottleneck: 模块输出特征（语义信息丰富）
        """
        if location == 'expansion':  # output of conv after act, same as block coutput
            # 激活后的特征信息（用于中间特征提取）
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            # 模块最终输出的特征信息
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x    # 保存残差连接
        x = self.conv(x)    # 动态卷积计算
        x = self.bn1(x)     # 批量归一化
        x = self.act1(x)    # 非线性激活

        # 条件残差连接
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)    # 应用DropPath正则化
            x += shortcut   # 添加残差连接
        return x


class GhostModule(nn.Module):
    """
    Ghost模块核心实现
    创新架构：
    primary_conv: 标准卷积生成基础特征（通道数=oup/ratio）
    cheap_operation: 深度卷积生成补充特征（通道数=oup*(ratio-1)/ratio）

    计算量分析：
    原始卷积计算量：H*W*inp*oup*k*k
    Ghost模块计算量：H*W*inp*m*k*k + H*W*m*(oup-m)*d*d
    其中 m = ceil(oup/ratio), d=dw_size（通常为3）
    当ratio=2时，计算量减少约50%
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act_layer=nn.ReLU, num_experts=4):
        super(GhostModule, self).__init__()
        self.oup = oup  # 输出通道数
        # 计算初始通道数（按比例压缩）
        init_channels = math.ceil(oup / ratio)
        # 计算廉价操作生成的通道数
        new_channels = init_channels * (ratio - 1)

        # 主要卷积操作（生成基础特征）
        self.primary_conv = nn.Sequential(
            # 1x1动态卷积（通道压缩）
            DynamicConv(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False, num_experts=num_experts),
            nn.BatchNorm2d(init_channels),
            act_layer() if act_layer is not None else nn.Sequential(),  # 可选激活
        )

        # 廉价操作（深度可分离卷积生成补充特征）
        self.cheap_operation = nn.Sequential(
            # 深度卷积（groups=init_channels实现通道分离）
            DynamicConv(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False,
                        num_experts=num_experts),
            nn.BatchNorm2d(new_channels),
            act_layer() if act_layer is not None else nn.Sequential(),
        )

    def forward(self, x):
        # 主要卷积路径
        x1 = self.primary_conv(x)
        # 廉价操作路径
        x2 = self.cheap_operation(x1)
        # 特征拼接 [B, init+new, H, W]
        out = torch.cat([x1, x2], dim=1)
        # 截取至目标通道数（处理整数计算误差）
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost瓶颈层（含可选SE注意力）
    三阶段处理：
    1. 特征扩展：GhostModule扩展通道数（增加模型容量）
    2. 深度卷积：空间特征提取（可选下采样）
    3. 特征压缩：GhostModule压缩通道数（减少计算量）

    特殊设计：
    - 步长>1时使用深度卷积下采样
    - 添加SE模块增强通道感知能力
    - 自适应捷径连接处理维度变化
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., drop_path=0., num_experts=4):
        super(GhostBottleneck, self).__init__()

        # 判断是否使用SE模块
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        mid_chs = in_chs * 2    # 中间层通道扩展

        # Point-wise expansion
        # 第一阶段Ghost模块（特征扩展）
        self.ghost1 = GhostModule(in_chs, mid_chs, act_layer=act_layer, num_experts=num_experts)

        # Depth-wise convolution
        # 深度卷积（步长>1时进行下采样）
        if self.stride > 1:
            # 深度卷积（分组数=输入通道数）
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)  # 无偏置
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        # SE注意力模块
        # 特殊处理：当act_layer为GELU时使用ReLU替代（兼容性）
        self.se = _SE_LAYER(mid_chs, se_ratio=se_ratio,
                            act_layer=act_layer if act_layer is not nn.GELU else nn.ReLU) if has_se else None

        # Point-wise linear projection
        # 第二阶段Ghost模块（特征压缩）
        self.ghost2 = GhostModule(mid_chs, out_chs, act_layer=None, num_experts=num_experts)

        # shortcut
        # 捷径连接（输入输出维度匹配时用恒等映射，否则用卷积调整）
        if in_chs == out_chs and self.stride == 1:
            # 恒等映射
            self.shortcut = nn.Sequential()
        else:
            # 维度调整模块
            self.shortcut = nn.Sequential(
                # 深度卷积下采样
                DynamicConv(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False, num_experts=num_experts),
                nn.BatchNorm2d(in_chs),
                # 1x1卷积调整通道
                DynamicConv(in_chs, out_chs, 1, stride=1, padding=0, bias=False, num_experts=num_experts),
                nn.BatchNorm2d(out_chs),
            )

        # DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 保留原始输出
        shortcut = x

        # --- 主路径 ---
        # 阶段1：特征扩展
        x = self.ghost1(x)

        # 阶段2：空间特征提取（下采样）
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # 阶段3：通道注意力
        if self.se is not None:
            x = self.se(x)

        # 阶段4：特征压缩
        x = self.ghost2(x)

        # --- 残差路径 ---
        # 合并路径 + DropPath
        x = self.shortcut(shortcut) + self.drop_path(x)
        return x


# ---------------------------- YOLO兼容模块 ----------------------------
class Bottleneck(nn.Module):
    """
    标准瓶颈块（YOLO基础模块）
    结构：Conv1 -> Conv2 -> 残差连接
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels 隐藏层通道数

        # 两个卷积层
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        # 残差连接条件
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # 残差连接或普通卷积
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    自动计算填充大小（保持输出尺寸不变）
    支持膨胀卷积计算
    示例：
    k=3,d=1 -> p=1
    k=3,d=2 -> k=5 -> p=2
    """
    if d > 1:
        # 调整有效内核大小（考虑膨胀率）
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size

    # 计算所需填充（保持尺寸不变）
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """标准卷积块（YOLO风格）
    组成：Conv2d + BatchNorm + SiLU
    特点：支持组卷积和膨胀卷积
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()

        # 卷积层（自动填充）
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        # 激活函数处理
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """标准前向传播"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合模式（用于模型部署优化）"""
        return self.act(self.conv(x))


class C2f(nn.Module):
    """
    高效CSP瓶颈结构（YOLOv8使用）
    创新点：梯度分流+特征重用
    处理流程：
    1. 输入特征图分割为两部分
    2. 主分支通过n个Bottleneck块
    3. 所有分支特征拼接输出
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # 初始卷积（输出通道=2*c）
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # 最终卷积（融合特征）
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        # 瓶颈块序列（n个）
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """前向传播（chunk实现）"""

        # 特征分割 [B, 2c, H, W] -> [B, c, H, W] * 2
        y = list(self.cv1(x).chunk(2, 1))

        # 主分支处理（n个Bottleneck）
        y.extend(m(y[-1]) for m in self.m)

        # 特征拼接 [y0, y1, m0(y1), m1(y1), ...]
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """备选前向传播（split实现）"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """
    CSP瓶颈结构（YOLOv5使用）
    与C2f的区别：双输入单路径处理
    处理流程：
    1. 两条独立卷积路径
    2. 主路径通过n个Bottleneck
    3. 双路径特征拼接
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 双路径卷积
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        # 融合卷积
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)

        # 瓶颈块序列
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """
    自定义卷积核的C3模块
    扩展能力：支持任意方形卷积核
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels

        # 替换为自定义核的Bottleneck
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class GhostC3k2(C2f):
    """
    融合Ghost模块的增强型CSP瓶颈（核心创新）
    设计特点：
    1. 继承C2f的高效特征重用架构
    2. 动态选择特征提取器：
       - c3k=True：使用大核C3k提取空间特征
       - c3k=False：使用GhostBottleneck轻量融合
    3. 平衡精度与效率

    适用场景：
    - 高精度需求：c3k=True（大核卷积）
    - 移动端部署：c3k=False（Ghost轻量）
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)

        # 动态模块选择
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else GhostBottleneck(self.c, self.c) for _ in range(n)
        )  # c3k = True用C3k大核空间提取特征, False 时用GhostBottleneck轻量融合特征


# ---------------------------- 测试代码 ----------------------------
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = GhostC3k2(64, 64)

    out = model(image)
    print(out.size())