import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.aspp import ASPPMSDA
import math
import warnings
from typing import List


# =======================================================
# 1. EGM-Style QKV Preprocessing (RefInputProj)
# =======================================================

class RefInputProj(nn.Module):
    """ 
    EGM风格的特征投影模块 (P, G 的预处理)。
    执行空间精炼 (DWConv) 和通道对齐 (1x1 Conv)。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            # 空间精炼: 深度可分离卷积 (DWConv)
            # groups=in_channels 实现了 depthwise 卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 通道对齐: 1x1 Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        )

    def forward(self, F_ref, F_curr_size):
        # F_ref: Reference Feature (P or G)
        # 1. 空间对齐 (Interpolation)
        if F_ref.shape[2:] != F_curr_size:
            F_ref = F.interpolate(F_ref, size=F_curr_size, mode='bilinear', align_corners=True)
            
        # 2. 投影和精炼
        return self.proj(F_ref)

# =======================================================
# 2. BDRB - Boundary Difference Residual Block (R_refine)
# =======================================================

class BDRB(nn.Module):
    """ 
    边界差分残差块，计算边界修正残差 R_refine。
    输入/输出通道均为 C_mvcm。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        # --- 边界差分层 (BDL) ---
        # BDL-A: 边界提取 (F_bound)
        #self.conv_bound_extract = nn.Conv2d(self.C, self.C, kernel_size=3, padding=1)
        self.conv_bound_extract = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            # 通道对齐: 1x1 Conv
            nn.Conv2d(self.C, self.C, kernel_size=1) 
        )
        
        # BDL-B/C: 降维聚焦与权重激活 (W_BDRB)
        self.conv_weight_generate = nn.Sequential(
            nn.Conv2d(self.C, 1, kernel_size=1), # 降至单通道
            nn.Sigmoid() 
        )
        
        # --- 残差路径 (Residual Path) ---
        
        # RP-A: 初始映射 (F_map)
        self.conv_map_1 = nn.Conv2d(self.C, self.C, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.C)
        
        # RP-D: 最终映射 (R_BDRB)
        self.conv_map_2 = nn.Conv2d(self.C, self.C, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.C)

    def forward(self, F_in):
        # 1. 边界权重生成 (BDL)
        F_bound = self.conv_bound_extract(F_in)
        W_BDRB = self.conv_weight_generate(F_bound)
        
        # 2. 残差路径 (RP)
        F_map = F.relu(self.bn1(self.conv_map_1(F_in)))
        
        # 边界加权 (Hadamard Product)
        F_weighted = F_map * W_BDRB 
        
        # 最终映射
        R_BDRB = self.bn2(self.conv_map_2(F_weighted))
        
        # 3. 跳跃连接 (F_in + R_BDRB)
        #R_refine = F_in + R_BDRB
        
        return R_BDRB


class CrossAttentionGuide(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # 线性映射层
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # 初始化权重 (可选，有助于稳定训练)
        nn.init.kaiming_normal_(self.to_q.weight)
        nn.init.kaiming_normal_(self.to_k.weight)
        nn.init.kaiming_normal_(self.to_v.weight)

    def forward(self, x_dest, x_src):
        """
        x_dest: 接收者 (Q) -> 被增强的流
        x_src:  引导者 (K, V) -> 提供信息的流
        """
        B, C, H, W = x_dest.shape
        
        # 1. 生成 Q, K, V
        q = self.to_q(x_dest)  # Query 来自 接收者
        k = self.to_k(x_src)   # Key   来自 引导者
        v = self.to_v(x_src)   # Value 来自 引导者
        
        # 2. 变换维度以适应 Multi-head 处理
        # [B, Heads, C/Heads, H*W]
        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)

        # 3. 计算注意力 (Pixel-wise / Spatial correspondence)
        # 注意：这里我们不做全图的 MatMul (H*W x H*W)，因为边缘需要严格的空间对齐。
        # 我们使用哈达玛积 (Hadamard product) 关注点对点的相关性，或者小范围的注意力。
        # 这里演示标准的 Dot-Product Attention (虽然计算量大，但物理意义最标准)
        
        # 方式 A: 标准 Global Attention (计算量大，适合小图)
        # attn = (q.transpose(-2, -1) @ k) * self.scale
        # attn = attn.softmax(dim=-1)
        # out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        
        # 方式 B: 局部/像素对齐 Attention (推荐用于边缘任务)
        # 逻辑：只计算对应位置的相似度，因为边缘和分割是像素对齐的
        attn_logit = (q * k).sum(dim=2, keepdim=True) * self.scale # [B, Heads, 1, HW]
        attn_map = attn_logit.softmax(dim=-1) # 注意：这里dim=-1是对空间归一化
        
        # 但对于 Guide 任务，通常 Sigmoid 更好，因为它代表"门控"而不是"分布"
        attn_gate = torch.sigmoid(attn_logit) 
        
        # 将 Value 注入
        out = v * attn_gate # 加权
        
        # 恢复形状
        out = out.view(B, C, H, W)
        out = self.proj(out)
        
        # 残差连接：原始特征 + 注入的引导信息
        return x_dest + out

class MVCMCollaborator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.msda_outref = ASPPMSDA(d_model=self.C, n_levels=3, n_heads=8, n_points=4)

        # 预处理 (Pre-fusion)
        self.pre_edge = nn.Sequential(nn.Conv2d(2*self.C, self.C, 3, 1, 1), nn.BatchNorm2d(self.C), nn.ReLU())

        # 交叉注意力模块
        self.attn_edge_to_seg = CrossAttentionGuide(self.C) # Edge 引导 Seg
        self.attn_seg_to_edge = CrossAttentionGuide(self.C) # Seg 引导 Edge

        self.final_fusion = nn.Conv2d(self.C, self.C, 3, 1, 1)

    def forward(self, F_curr, F_outref_list):
        F_skip = F_curr

        Fn_outref = self.msda_outref(F_curr, F_outref_list)

        # 1. 初始化双流
        # Seg流: 直接使用当前特征
        F_seg = F_curr
        # Edge流主要看 Fn_outref (边缘)
        F_edge = self.pre_edge(torch.cat([F_curr, Fn_outref], dim=1))

        # 2. 交互 Step 1: Seg 引导 Edge (去噪)
        # Q=Edge, K=Seg, V=Seg
        # "Edge流，请根据Seg流的信息，把背景噪声去掉"
        F_edge_refined = self.attn_seg_to_edge(x_dest=F_edge, x_src=F_seg)

        # 3. 交互 Step 2: Edge 引导 Seg (锐化)
        # Q=Seg, K=Edge_refined, V=Edge_refined
        # "Seg流，请根据净化后的Edge流，增强边界"
        # 注意：这里用的是刚刚净化过的 F_edge_refined
        F_seg_refined = self.attn_edge_to_seg(x_dest=F_seg, x_src=F_edge_refined)

        out = self.final_fusion(F_seg_refined)
        return F_skip + out

# =======================================================
# 4. CRFB（修改：只使用边缘特征进行多尺度融合）
# =======================================================
class CRFB(nn.Module):
    def __init__(self, C_enc, C_g_ref, C_mvcm):
        super().__init__()
        self.proj_curr = nn.Conv2d(C_enc, C_mvcm, 1)
        self.proj_outref = RefInputProj(C_g_ref, C_mvcm)
        self.mvcm_collaborator = MVCMCollaborator(in_channels=C_mvcm)
        self.bdrb = BDRB(in_channels=C_mvcm)

    def _create_multiscale(self, F_ref, target_size):
        B, C, H, W = F_ref.shape
        Ht, Wt = target_size

        # 1x
        f1 = F.interpolate(F_ref, size=(Ht, Wt), mode='bilinear', align_corners=True)

        # 1/2x
        H2, W2 = max(Ht // 2, 1), max(Wt // 2, 1)
        f05 = F.interpolate(F_ref, size=(H2, W2), mode='bilinear', align_corners=True)

        # 2x
        H4, W4 = min(Ht * 2, H * 4), min(Wt * 2, W * 4)
        f2 = F.interpolate(F_ref, size=(H4, W4), mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, size=(Ht, Wt), mode='bilinear', align_corners=True)

        return [f05, f1, f2]  # low, mid, high

    def forward(self, F_curr_enc, F_edge, F_next_enc):
        F_curr = self.proj_curr(F_curr_enc)
        target_size = F_curr.shape[2:]

        # 自动生成多尺度 OutRef
        F_outref_list = self._create_multiscale(self.proj_outref(F_edge, target_size), target_size)

        # MVCM 融合
        F_CRFB = self.mvcm_collaborator(F_curr, F_outref_list)

        # BDRB 细化
        F_CRFB_up = F.interpolate(F_CRFB, size=F_next_enc.shape[2:], mode='bilinear', align_corners=True)
        R_refine = self.bdrb(F_CRFB_up)
        F_next_input = F_next_enc + R_refine

        return F_next_input, F_CRFB
