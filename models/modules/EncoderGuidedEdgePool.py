"""
EncoderGuidedEdgePool: 利用编码器语义引导进行边缘节点筛选和去噪

原理:
  Edge features (px_edge) + Encoder features (semantic guidance)
           ↓
  importance = ||px_edge|| × sigmoid(encoder_activation)
           ↓
  Top-K selection → Keep significant edges, discard background noise
           ↓
  Interpolate back to original resolution

特点:
  - 无需构建图结构，简单轻量
  - 利用编码器的语义响应来指导边缘节点的重要性评估
  - Top-K 筛选去除背景边缘噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGuidedEdgePool(nn.Module):
    """
    编码器引导的边缘池化模块

    输入:
      edge_feat: 边缘特征 [B, C_mvcm, H, W]
      encoder_feat: 对应尺度的编码器特征 [B, C_enc, H, W]

    输出:
      enhanced_edge: 去噪后的边缘特征 [B, C_mvcm, H, W]
    """
    def __init__(self, edge_channels, encoder_channels, c_mvcm=64, keep_ratio=0.25):
        super().__init__()
        self.keep_ratio = keep_ratio

        # 边缘特征重要性评分网络
        self.edge_importance_net = nn.Sequential(
            nn.Conv2d(edge_channels, c_mvcm, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mvcm, 1, 1),
            nn.Sigmoid()
        )

        # 编码器激活度网络 (将编码器特征转换为 0~1 的响应)
        self.encoder_activation_net = nn.Sequential(
            nn.Conv2d(encoder_channels, c_mvcm, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mvcm, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_feat, encoder_feat):
        """
        edge_feat: [B, C_edge, H, W] - 边缘特征
        encoder_feat: [B, C_enc, H, W] - 编码器特征 (语义引导)

        Returns: enhanced_edge [B, C_edge, H, W]
        """
        B, C_edge, H, W = edge_feat.shape

        # 1. 计算边缘重要性: ||px_edge|| × sigmoid(encoder_activation)
        edge_imp = self.edge_importance_net(edge_feat)  # [B, 1, H, W]
        encoder_act = self.encoder_activation_net(encoder_feat)  # [B, 1, H, W]
        importance = edge_imp * encoder_act  # [B, 1, H, W]

        # 2. 展平并执行 Top-K 筛选
        importance_flat = importance.view(B, -1)  # [B, H*W]
        k = max(1, int(H * W * self.keep_ratio))
        _, topk_idx = torch.topk(importance_flat, k=k, dim=1)  # [B, K]

        # 3. 收集 Top-K 节点
        edge_flat = edge_feat.view(B, C_edge, -1)  # [B, C_edge, H*W]
        topk_idx_expanded = topk_idx.unsqueeze(1).expand(-1, C_edge, -1)  # [B, C_edge, K]
        pooled_edge = torch.gather(edge_flat, dim=2, index=topk_idx_expanded)  # [B, C_edge, K]

        # 4. 重建: 将 Top-K 节点散布回原始分辨率
        # 创建输出张量 [B, C_edge, H*W]，初始化为 0
        output_flat = torch.zeros(B, C_edge, H * W, device=edge_feat.device, dtype=edge_feat.dtype)
        # 将 Top-K 节点放回对应位置
        output_flat.scatter_(dim=2, index=topk_idx_expanded, src=pooled_edge)

        # 计算每个位置被填充的次数 (用于归一化)
        count_mask = torch.zeros(B, H * W, device=edge_feat.device)
        count_mask.scatter_(dim=1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=count_mask.dtype))

        # 归一化: 每个位置的值 / 该位置被选中的次数
        output_flat = output_flat / count_mask.unsqueeze(1).clamp(min=1)

        # reshape 回 [B, C_edge, H, W]
        enhanced_edge = output_flat.view(B, C_edge, H, W)

        return enhanced_edge


class MultiScaleEncoderGuidedEdgePool(nn.Module):
    """
    多尺度编码器引导边缘池化

    对 x1, x2, x3, x4 四个尺度的边缘特征分别进行去噪
    """
    def __init__(self, encoder_channels_list, edge_channels=64, c_mvcm=64, keep_ratios=[0.2, 0.4, 0.6, 0.8]):
        super().__init__()
        """
        encoder_channels_list: [192, 384, 768, 1536] - x1,x2,x3,x4 的通道数 (swin_v1_l)
        edge_channels: 边缘特征通道数 (默认 64)
        c_mvcm: 处理通道数 (默认 64)
        keep_ratios: 每层保留比例 [x1, x2, x3, x4]
        """
        self.pool_list = nn.ModuleList([
            EncoderGuidedEdgePool(
                edge_channels=edge_channels,
                encoder_channels=enc_ch,
                c_mvcm=c_mvcm,
                keep_ratio=keep_ratios[i]
            )
            for i, enc_ch in enumerate(encoder_channels_list)
        ])

    def forward(self, encoder_feats, edge_feats):
        """
        encoder_feats: [x1, x2, x3, x4] - 编码器多尺度特征 (从高分辨率到低分辨率)
        edge_feats: [px1_edge, px2_edge, px3_edge, px4_edge] - 多尺度边缘特征

        Returns:
          enhanced_edges: [enhanced_px1_edge, enhanced_px2_edge, enhanced_px3_edge, enhanced_px4_edge]
        """
        enhanced_edges = []
        for i in range(len(encoder_feats)):
            # 对齐边缘特征到编码器特征的分辨率
            aligned_edge = F.interpolate(
                edge_feats[i],
                size=encoder_feats[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )
            enhanced = self.pool_list[i](aligned_edge, encoder_feats[i])
            enhanced_edges.append(enhanced)

        return enhanced_edges
