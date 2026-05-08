import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import laplacian
from huggingface_hub import PyTorchModelHubMixin
from config import Config
from dataset import class_labels_TR_sorted
from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk, ResBlk
from models.modules.lateral_blocks import BasicLatBlk
from models.modules.aspp import ASPP, ASPPDeformable
from models.refinement.refiner import Refiner, RefinerPVTInChannels4, RefUNet
from models.refinement.stem_layer import StemLayer
# === 新增导入用于 MSD-CoT 集成 ===
from PIL import Image
from torchvision import transforms 
from .utils_msd_cot import crop_image_with_bbox, crop_image_with_centerpoint 
# =================================
import cv2
from models.modules.CRFB import CRFB
from models.modules.EncoderGuidedEdgePool import MultiScaleEncoderGuidedEdgePool 

# 新增：导入小波变换依赖（若未安装需先执行pip install pywavelets）
import pywt
import numpy as np

# ---------------------- 1. 新增/修改小波变换核心模块（含IDWT） ----------------------
class DWT_2D(nn.Module):
    """二维离散小波变换，输出LL/LH/HL/HH四个子带（用于分解）"""
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        wavelet = pywt.Wavelet(wavelet)
        dec_lo, dec_hi = wavelet.dec_lo, wavelet.dec_hi
        # 构造2D分解卷积核（水平×垂直）
        self.kernel_ll = torch.tensor(np.outer(dec_lo, dec_lo).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.kernel_lh = torch.tensor(np.outer(dec_lo, dec_hi).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.kernel_hl = torch.tensor(np.outer(dec_hi, dec_lo).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.kernel_hh = torch.tensor(np.outer(dec_hi, dec_hi).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # 注册固定卷积核（小波基不参与训练，保留结构特性）
        self.register_buffer('weight_ll', self.kernel_ll)
        self.register_buffer('weight_lh', self.kernel_lh)
        self.register_buffer('weight_hl', self.kernel_hl)
        self.register_buffer('weight_hh', self.kernel_hh)

    def forward(self, x):
        """输入x(B,C,H,W)，输出LL/LH/HL/HH(B,C,H//2,W//2)"""
        B, C, H, W = x.shape
        # 分组卷积：每个通道独立分解，避免通道干扰
        ll = F.conv2d(x, self.weight_ll.repeat(C,1,1,1), stride=2, padding=0, groups=C)
        lh = F.conv2d(x, self.weight_lh.repeat(C,1,1,1), stride=2, padding=0, groups=C)
        hl = F.conv2d(x, self.weight_hl.repeat(C,1,1,1), stride=2, padding=0, groups=C)
        hh = F.conv2d(x, self.weight_hh.repeat(C,1,1,1), stride=2, padding=0, groups=C)
        return ll, lh, hl, hh


class WaveletSubbandProcessor(nn.Module):
    """小波子带处理器：LL+IDWT重构增强InRef结构，高频聚合增强OutRef细节"""
    def __init__(self, in_channels_ll, in_channels_high, out_channels_ll, out_channels_high, wavelet='haar',use_high=True):
        super().__init__()
        self.use_high = use_high
        self.wavelet = wavelet
        # 1. IDWT重构后LL子带的通道调整（匹配InRef补丁特征通道）
        # self.ll_extractor = nn.Sequential(
        #     nn.Conv2d(in_channels_ll, out_channels_ll // 4, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels_ll // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels_ll // 4, out_channels_ll, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels_ll),
        #     nn.ReLU(inplace=True)
        # )
        #self.inref_pro = nn.Conv2d(in_channels_ll, out_channels_ll, kernel_size=3, padding=1, bias=False)

        # 2. 调制参数生成器 (生成 Scale 和 Shift)
        # 类似于 SPADE 或 FiLM 模块
        #self.gamma_generator = nn.Conv2d(out_channels_ll, out_channels_ll, kernel_size=1) # Scale
        #self.beta_generator = nn.Conv2d(out_channels_ll, out_channels_ll, kernel_size=1)  # Shift
        
        # 3. 融合后的平滑层 (可选)
        # self.post_fusion = nn.Sequential(
        #     nn.Conv2d(out_channels_ll, out_channels_ll, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels_ll),
        #     nn.ReLU(inplace=True)
        # )

        # 2. 高频子带（LH+HL+HH）聚合（匹配OutRef梯度特征通道）
        if use_high:
            self.high_aggregator = nn.Sequential(
                nn.Conv2d(in_channels_high * 3, out_channels_high, 3, 1, 1),  # 三高频拼接融合
                nn.BatchNorm2d(out_channels_high) if out_channels_high > 1 else nn.Identity(),
                nn.ReLU(inplace=True)
            )

    # def process_ll(self, ll_subband, inref_patch):
    #     """
    #     利用 LL 子带的全局结构信息，动态调制 InRef 的局部细节
    #     """
    #     # 1. 对齐尺寸与特征提取
    #     # ll_subband 通常较小，先上采样
    #     target_size = inref_patch.shape[2:]
    #     inref_patch = self.inref_pro(inref_patch)
    #     ll_upsampled = F.interpolate(ll_subband, size=target_size, mode='bilinear', align_corners=True)
        
    #     # 提取 LL 的结构特征
    #     ll_feat = self.ll_extractor(ll_upsampled) # [B, C, H, W]

    #     # 2. 生成调制参数
    #     # Gamma (缩放因子): 决定特征的活跃程度
    #     gamma = self.gamma_generator(ll_feat) 
    #     # Beta (平移因子): 提供背景偏置
    #     beta = self.beta_generator(ll_feat)

    #     # 3. 执行仿射变换 (Modulation)
    #     # 公式: Out = InRef * (1 + Gamma) + Beta
    #     # (1 + Gamma) 类似于残差缩放，保证初始状态下 InRef 不会消失
    #     fused_out = inref_patch * (1 + gamma) + beta
        
    #     # 4. 后处理 (融合特征整合)
    #     out = self.post_fusion(fused_out)
        
    #     # 5. 残差连接 (保留原始细节流的稳定性)
    #     return inref_patch + out

    def process_high(self, lh_subband, hl_subband, hh_subband):
        if not self.use_high:
            return None
        """高频子带聚合：增强边缘、纹理细节（用于匹配OutRef梯度特征）"""
        # 三高频子带通道拼接
        high_concat = torch.cat([lh_subband, hl_subband, hh_subband], dim=1)
        # 多方向边缘融合
        high_aggregated = self.high_aggregator(high_concat)
        return high_aggregated


def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation='b c (hg h) (wg w) -> (b hg wg) c h w'):
    if patch_ref is not None:
        grid_h, grid_w = image.shape[-2] // patch_ref.shape[-2], image.shape[-1] // patch_ref.shape[-1]
        # grid_h * grid_w是补丁的数量
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    return patches    #[N,3072,16,16] [N,1536,32,32] [N,768,64,64] [N,384,128,128] [N,3,512,512]
    # grid_h=grid_w=32

  
class SimpleConvs(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, inter_channels=64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


class BiRefNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet",
    repo_url="https://github.com/ZhengPeng7/BiRefNet",
    tags=['Image Segmentation', 'Background Removal', 'Mask Generation', 'Dichotomous Image Segmentation',
          'Camouflaged Object Detection', 'Salient Object Detection']
):
    def __init__(self, bb_pretrained=True):
        super(BiRefNet, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.bb = build_backbone(self.config.bb, pretrained=bb_pretrained)

        channels = self.config.lateral_channels_in_collection

        if self.config.auxiliary_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], len(class_labels_TR_sorted))
            )

        if self.config.squeeze_block:
            self.squeeze_module = nn.Sequential(*[
                eval(self.config.squeeze_block.split('_x')[0])(channels[0] + sum(self.config.cxt), channels[0])
                for _ in range(eval(self.config.squeeze_block.split('_x')[1]))
            ])
        
        # --- 修改: Decoder 初始化现在需要额外的通道信息 ---
        C_grad_ref = 64 # 梯度 G 仍保持 3 通道 (或 1)
        C_mvcm = 64
        self.decoder = Decoder(channels, C_grad_ref=C_grad_ref, C_mvcm=C_mvcm)
        # ----------------------------------------  

        if self.config.ender:
            self.dec_end = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.Conv2d(16, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        # refine patch-level segmentation
        if self.config.refine:
            if self.config.refine == 'itself':
                self.stem_layer = StemLayer(in_channels=3 + 1, inter_channels=48, out_channels=3,
                                            norm_layer='BN' if self.config.batch_size > 1 else 'LN')
            else:
                self.refiner = eval('{}({})'.format(self.config.refine, 'in_channels=3+1'))

        if self.config.freeze_bb:
            # Freeze the backbone...
            print(self.named_parameters())
            for key, value in self.named_parameters():
                if 'bb.' in key and 'refiner.' not in key:
                    value.requires_grad = False

    def forward_enc(self, x):
        if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:  # self.config.bb = swim_v1_l
            x1 = self.bb.conv1(x);
            x2 = self.bb.conv2(x1);
            x3 = self.bb.conv3(x2);
            x4 = self.bb.conv4(x3)
        else:
            x1, x2, x3, x4 = self.bb(x)  # x1[N, 192, 128, 128] x2[N, 384, 64, 64]  x3[N, 768, 32, 32] x4[N, 1536, 16, 16]
        if self.config.mul_scl_ipt:  # cat
            B, C, H, W = x.shape  # 获得原始特征图的B=4，C=3，H=512，W=512
            x_pyramid = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True) #x_pyramid[N,3,256,256]
            if self.config.mul_scl_ipt == 'cat':
                if self.config.bb in ['vgg16', 'vgg16bn', 'resnet50']:
                    x1_ = self.bb.conv1(x_pyramid);
                    x2_ = self.bb.conv2(x1_);
                    x3_ = self.bb.conv3(x2_);
                    x4_ = self.bb.conv4(x3_)
                else:
                    x1_, x2_, x3_, x4_ = self.bb(x_pyramid) # x1_[N, 192, 64, 64] x2_[N, 384,32, 32] x3_[N, 768, 16, 16] x4_[N, 1536, 8, 8]
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1) #[N,384,128,128]
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1) #[N,768,64,64]
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1) #[N,1536,32,32]
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1) #[N,3072,16,16]
            elif self.config.mul_scl_ipt == 'add':
                x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)
        class_preds = self.cls_head(
            self.avgpool(x4).view(x4.shape[0], -1)) if self.training and self.config.auxiliary_classification else None #COD:None
        if self.config.cxt:  # self.config.cxt[0=384,1=768,2=1536]
            x4 = torch.cat(
                (
                    *[
                         F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),  # 先将x1插值到与x4相同分辨率
                         F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                     ][-len(self.config.cxt):],
                    x4
                ),
                dim=1
            )
        return (x1, x2, x3, x4), class_preds  # x1[N,384,128,128] x2[N,768,64,64],x3[N,1536,32,32] x4[N,5760,16,16] class_preds=None
        

    def forward_ori(self, x, bboxes_points):  # x=[1,3,512,512]

        ########## Encoder ##########
        (x1, x2, x3, x4), class_preds = self.forward_enc(x) # # x1[N,384,128,128] x2[N,768,64,64],x3[N,1536,32,32] x4[N,5760,16,16] class_preds=None
        if self.config.squeeze_block:  # self.config.squeeze_block=BasicDecBlk_x1
            x4 = self.squeeze_module(x4)  # x4[N,3072,16,16]
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]   #x[N,3,512,512] x1[N,384,128,128] x2[N,768,64,64] x3[N,1536,32,32] x4[N,3072,16,16] 原始图像 + Swim Transformer生成的多尺度特征
        #if self.training and self.config.out_ref:  # True 
        #    features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))  # 如果有外参考就把原始图像经过拉普拉斯梯度[N,1,512,512]
        scaled_preds = self.decoder(features, bboxes_points) # **修改：传入 bboxes_points**
        return scaled_preds, class_preds

    
    def forward(self, x, bboxes_points):
        scaled_preds, class_preds = self.forward_ori(x, bboxes_points)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds


class FocusCannyEdgeModule(nn.Module):
    """
    FOCUS风格边缘提取模块（可嵌入PyTorch模型）
    适配网络训练流水线，支持forward调用，无梯度回传
    """
    def __init__(self, threshold1=80, threshold2=160):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    @torch.no_grad()  # 禁用梯度计算，完全对齐FOCUS无梯度特性
    def forward(self, x):
        """
        Args:
            x: 输入RGB张量，shape [B, 3, H, W]，值范围[0, 1]
        Returns:
            edge: 边缘特征张量，shape [B, 1, H, W]，值范围[0, 1]
        """
        B, C, H, W = x.shape
        x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        x_np = (x_np * 255).astype(np.uint8)
        
        edges_np = []
        for img in x_np:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            canny_edge = cv2.Canny(gray, self.threshold1, self.threshold2)
            edges_np.append(canny_edge[:, :, np.newaxis])
        
        edge = torch.tensor(np.array(edges_np), dtype=torch.float32)
        edge = edge.permute(0, 3, 1, 2) / 255.0
        return edge.cuda() if x.is_cuda else edge


class AdaptiveGradientModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.C = out_channels

        # H 方向分支：独立从 x 提取，加 BN+ReLU
        self.conv1x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.re1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # V 方向分支：独立从 x 提取（非串行），加 BN+ReLU
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.re2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # 对角线方向分支（D1: \, D2: /）：用 depthwise 3x3 提取
        self.conv_diag = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.re_d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # 多尺度融合：dilate=2 感受野 5x5
        self.conv3x3dil = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.re3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # H 和 V 分支独立提取（解决方向串扰）
        add_h = self.conv1x3(x) + self.re1(x)  # 水平边缘
        add_v = self.conv3x1(x) + self.re2(x)  # 垂直边缘（从 x 而非 add_h）
        add_d = self.conv_diag(x) + self.re_d(x)  # 对角线边缘

        # 融合三个方向 + 多尺度
        fused = add_h + add_v + add_d
        add3 = self.conv3x3dil(fused) + self.re3(fused)

        return self.project(add3)
    

class UncertaintyGatedRefiner(nn.Module):
    def __init__(self, in_channels, high_freq_channels):
        super().__init__()
        # 用于生成初步预测，从而计算不确定度
        self.coarse_head = nn.Conv2d(high_freq_channels, 1, 1)
        
        # 高频特征压缩与整合
        self.high_compress = nn.Sequential(
            nn.Conv2d(in_channels, high_freq_channels, 3, 1, 1),
            nn.BatchNorm2d(high_freq_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Conv2d(in_channels + high_freq_channels, in_channels * 2, 1)

        # 融合后的特征整合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_base, f_high):
        # 1. 计算不确定度图 (Uncertainty Map)
        # 越接近0.5，值越大 (最高为1.0)
        p = torch.sigmoid(self.coarse_head(f_high))
        uncertainty = 1.0 - torch.abs(2.0 * p - 1.0)
        
        # 2. 只有在不确定区域，高频特征才被激活
        f_high_gated = self.high_compress(f_base) * uncertainty
        
        # 3. 拼接并融合
        out = self.project(torch.cat([f_base, f_high_gated], dim=1))
        out = self.fusion(out)

        return out
    

class Decoder(nn.Module):
    # **修改：forward 接收 bboxes_points**
    def __init__(self, channels: list, C_grad_ref: int, C_mvcm: int):
        super(Decoder, self).__init__()
        self.config = Config()
        #DecoderBlock = eval(self.config.dec_blk)
        #LateralBlock = eval(self.config.lat_blk)
        # 编码器通道 [C4, C3, C2, C1]
        self.C_enc = channels 
        self.C_grad_ref = C_grad_ref
        self.C_mvcm = C_mvcm
        self.out_channels_high = 32
        self.total_high_c = self.C_mvcm + self.out_channels_high

        #if self.config.dec_ipt:
        self.split = self.config.dec_ipt_split
        N_dec_ipt = 64
        #DBlock = SimpleConvs
        ic = 64
        ipt_cha_opt = 1
        ipt_blk_in_channels = [2**i*3 for i in (10, 8, 6, 4)] if self.split else [3] * 5  # [3072, 768, 192, 48]
        ipt_blk_out_channels = [[N_dec_ipt, channels[i]//8][ipt_cha_opt] for i in range(3)]  # [384, 192, 96] 
        # 这里应该是将图像块编码为特征
        # 修复错误：将输入通道从 3072/1536/... 改回 3 (因为输入是裁剪后的 3 通道图像)
        # self.ipt_blk5 = DBlock(3, ipt_blk_out_channels[0], inter_channels=ic)
        # self.ipt_blk4 = DBlock(3, ipt_blk_out_channels[0], inter_channels=ic)
        # self.ipt_blk3 = DBlock(3, ipt_blk_out_channels[1], inter_channels=ic)
        # self.ipt_blk2 = DBlock(3, ipt_blk_out_channels[2], inter_channels=ic)
        #self.ipt_blk1 = DBlock(3, [N_dec_ipt, channels[3] // 8][ipt_cha_opt], inter_channels=ic)
        # else:
        #     self.split = None

        self.lt_refiner4 = UncertaintyGatedRefiner(in_channels=self.C_mvcm, high_freq_channels=self.total_high_c)
        self.lt_refiner3 = UncertaintyGatedRefiner(in_channels=self.C_mvcm, high_freq_channels=self.total_high_c)
        self.lt_refiner2 = UncertaintyGatedRefiner(in_channels=self.C_mvcm, high_freq_channels=self.total_high_c)

        self.image_edge_extractor = nn.Sequential(
            AdaptiveGradientModule(3, 32),
            nn.Conv2d(32, C_mvcm, 3, 1, 1),
            # nn.BatchNorm2d(C_mvcm),
            # nn.ReLU(inplace=True)
        )

        # 1. 每一层的边缘特征提取 (从 CRFB 输出中提取边缘特征)
        self.edge_convs = nn.ModuleList([
            AdaptiveGradientModule(C_mvcm, C_mvcm)
            for i in range(3)
        ])

        # 2. 每一层的边缘预测头 (输出 1-channel 用于计算 Loss)
        self.edge_pred_heads = nn.ModuleList([
            nn.Conv2d(C_mvcm, 1, 1) for _ in range(4)
        ])

        # === EncoderGuidedEdgePool: 编码器引导的边缘去噪模块 ===
        # 对 x1, x2, x3, x4 四个尺度的边缘特征进行语义引导去噪
        # keep_ratios: [0.2, 0.4, 0.6, 0.8] - 从高分辨率到低分辨率保留更多节点
        self.edge_pool = MultiScaleEncoderGuidedEdgePool(
            encoder_channels_list=self.C_enc[::-1],  # [C4, C3, C2, C1] -> [x4, x3, x2, x1]
            edge_channels=C_mvcm,
            c_mvcm=C_mvcm,
            keep_ratios=[0.2, 0.4, 0.6, 0.8]
        )

        self.canny_gen = FocusCannyEdgeModule(threshold1=80, threshold2=160)

        # --- 1. CRFB 模块初始化 ---
        # CRFB4: C_enc=C4, C_g_ref=C_G
        self.crfb4 = CRFB(C_enc=self.C_enc[0], C_g_ref=C_grad_ref, C_mvcm=C_mvcm)
        # CRFB3: C_enc=C_mvcm, C_g_ref=C_G
        self.crfb3 = CRFB(C_enc=C_mvcm, C_g_ref=C_grad_ref, C_mvcm=C_mvcm)
        # CRFB2: C_enc=C_mvcm, C_g_ref=C_G
        self.crfb2 = CRFB(C_enc=C_mvcm, C_g_ref=C_grad_ref, C_mvcm=C_mvcm)
        # CRFB1: C_enc=C_mvcm, C_g_ref=C_G
        self.crfb1 = CRFB(C_enc=C_mvcm, C_g_ref=C_grad_ref, C_mvcm=C_mvcm)

        # --- 2. BDRB 残差连接的编码器投影层 (将 F_e^i 投影到 C_mvcm) ---
        self.proj_e3 = nn.Conv2d(self.C_enc[1], C_mvcm, 1) 
        self.proj_e2 = nn.Conv2d(self.C_enc[2], C_mvcm, 1) 
        self.proj_e1 = nn.Conv2d(self.C_enc[3], C_mvcm, 1)
        self.proj_e0 = nn.Conv2d(3, C_mvcm, 1)

        # --- 3. 最终预测头和辅助监督头 ---
        #total_channels = C_mvcm + ipt_blk_out_channels[2]
        #self.final_pred_head = nn.Conv2d(total_channels, 1, 1)
        self.final_pred_head = nn.Conv2d(C_mvcm, 1, 1)
        self.aux_pred_heads = nn.ModuleList([
            nn.Conv2d(C_mvcm, 1, 1) for _ in range(3) 
        ])

        #---------------------- 新增2：初始化小波变换与子带处理器 ----------------------
        self.dwt = DWT_2D(wavelet='haar')  # 选择Haar小波（轻量，计算快）
        # 配置LL子带处理器参数（InRef补丁输入通道=3，输出通道=ipt_blk_out_channels，匹配原补丁特征）
        self.wavelet_processor5 = WaveletSubbandProcessor(
            in_channels_ll=3,  # LL子带输入：原始图像补丁（RGB，3通道）
            in_channels_high=3,  # 高频子带输入：原始图像补丁（RGB，3通道）
            out_channels_ll=ipt_blk_out_channels[0],  # 输出通道=原InRef补丁处理后通道）
            out_channels_high=self.out_channels_high,  # 输出通道=OutRef梯度特征通道（匹配self.gdt_convs_4的输出16）
            use_high=True
        )

        self.wavelet_processor4 = WaveletSubbandProcessor(
            in_channels_ll=3,  # LL子带输入：原始图像补丁（RGB，3通道）
            in_channels_high=3,  # 高频子带输入：原始图像补丁（RGB，3通道）
            out_channels_ll=ipt_blk_out_channels[0],  # 输出通道=原InRef补丁处理后通道）
            out_channels_high=self.out_channels_high,  # 输出通道=OutRef梯度特征通道（匹配self.gdt_convs_4的输出16）
            use_high=True
        )

        self.wavelet_processor3 = WaveletSubbandProcessor(
            in_channels_ll=3,  # LL子带输入：原始图像补丁（RGB，3通道）
            in_channels_high=3,  # 高频子带输入：原始图像补丁（RGB，3通道）
            out_channels_ll=ipt_blk_out_channels[1],  # 输出通道=原InRef补丁处理后通道
            out_channels_high=self.out_channels_high,  # 输出通道=OutRef梯度特征通道（匹配self.gdt_convs_4的输出16）
            use_high=True
        )

        # self.wavelet_processor2 = WaveletSubbandProcessor(
        #     in_channels_ll=3,  # LL子带输入：原始图像补丁（RGB，3通道）
        #     in_channels_high=3,  # 高频子带输入：原始图像补丁（RGB，3通道）
        #     out_channels_ll=ipt_blk_out_channels[2],  # 输出通道=原InRef补丁处理后通道
        #     out_channels_high=self.out_channels_high,  # 输出通道=OutRef梯度特征通道（匹配self.gdt_convs_4的输出16）
        #     use_high=False
        # )

        # self.wavelet_processor1 = WaveletSubbandProcessor(
        #     in_channels_ll=3,  # LL子带输入：原始图像补丁（RGB，3通道）
        #     in_channels_high=3,  # 高频子带输入：原始图像补丁（RGB，3通道）
        #     out_channels_ll=ipt_blk_out_channels[3],  # 输出通道=原InRef补丁处理后通道
        #     out_channels_high=16,  # 输出通道=OutRef梯度特征通道（匹配self.gdt_convs_4的输出16）
        #     use_high=False
        # )


    def forward(self, features, bboxes_points): # **修改：添加 bboxes_points 参数**
        # if self.training and self.config.out_ref:
        #     outs_gdt_pred = [] 
        #     outs_gdt_label = []
        #     x, x1, x2, x3, x4, gdt_gt = features
        # else:
        #     x, x1, x2, x3, x4 = features  # 如果没有外参考就没有梯度标签
        # outs = []

        x, x1, x2, x3, x4 = features
        B, _, H, W = x.shape

        #G_raw = laplacian(torch.mean(x, dim=1, keepdim=True), kernel_size=5)
        with torch.no_grad():
            canny_gt = self.canny_gen(x) # [B, 1, H, W]

        seg_preds = []
        if self.training:
            outs_edge_pred = []  # 存储各层边缘预测
            outs_edge_label = [] # 存储各层过滤后的边缘真值

        if self.config.dec_ipt:  # true
            # **修改：替换 image2patches 逻辑为 BBOX 裁剪**
            # bbox_cropped_feature = crop_image_with_bbox(x, bboxes_points, target_size=x4.shape[2:]) 
            #P4 = crop_image_with_centerpoint(x, bboxes_points, target_size=x4.shape[2:]) 
            #P4 = F.interpolate(P4, size=x4.shape[2:], mode='bilinear', align_corners=True)
            #t = self.ipt_blk5(P4)
            pure_image_patch = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)  # (B,3,16,16)
            _, lh4, hl4, hh4 = self.dwt(pure_image_patch)
            #P4_fused = self.wavelet_processor5.process_ll(ll_subband=ll_subband, inref_patch=P4)
            high_wavelet_4 = self.wavelet_processor5.process_high(lh4, hl4, hh4) # [B, 16, H/32, W/32]

        # === 预提取边缘特征并应用 EncoderGuidedEdgePool 去噪 ===
        # 提取四个尺度的边缘特征 (从高分辨率到低分辨率)
        px4_edge_high_res = self.image_edge_extractor(x)
        px1_edge_pre = F.interpolate(px4_edge_high_res, size=x1.shape[2:], mode='bilinear', align_corners=True)
        px2_edge_pre = F.interpolate(px4_edge_high_res, size=x2.shape[2:], mode='bilinear', align_corners=True)
        px3_edge_pre = F.interpolate(px4_edge_high_res, size=x3.shape[2:], mode='bilinear', align_corners=True)
        px4_edge_pre = F.interpolate(px4_edge_high_res, size=x4.shape[2:], mode='bilinear', align_corners=True)

        # 应用编码器引导的边缘去噪 (语义引导: 利用编码器特征筛选重要边缘节点)
        # 输入: encoder_feats=[x1,x2,x3,x4], edge_feats=[px1,px2,px3,px4] (按分辨率从高到低)
        # 输出: enhanced_edges[px1_enh, px2_enh, px3_enh, px4_enh]
        enhanced_edges = self.edge_pool(
            encoder_feats=[x1, x2, x3, x4],
            edge_feats=[px1_edge_pre, px2_edge_pre, px3_edge_pre, px4_edge_pre]
        )
        px1_edge_denoised, px2_edge_denoised, px3_edge_denoised, px4_edge_denoised = enhanced_edges

        px4_edge = px4_edge_denoised

        F_next_enc_proj = self.proj_e3(x3) 
        F_next_input_3, F_CRFB_4 = self.crfb4(x4, px4_edge, F_next_enc_proj)
        px4_edge = F.interpolate(px4_edge, size=F_next_input_3.shape[2:], mode='bilinear', align_corners=True)
        high_wavelet_4 = F.interpolate(high_wavelet_4, size=F_next_input_3.shape[2:], mode='bilinear', align_corners=True)
        detail_packet_4 = torch.cat([high_wavelet_4, px4_edge], dim=1)
        F_next_3_refined = self.lt_refiner4(F_next_input_3, detail_packet_4)

        M4_up = F.interpolate(self.aux_pred_heads[0](F_CRFB_4), size=x.shape[2:], mode='bilinear', align_corners=True)
        seg_preds.append(M4_up) # M4

        if self.training and self.config.out_ref:
            edge_pred_4 = self.edge_pred_heads[0](px4_edge_high_res)
            outs_edge_pred.append(edge_pred_4)
            outs_edge_label.append(canny_gt * torch.sigmoid(M4_up.detach()))

        if self.config.dec_ipt:
            # **修改：替换 image2patches 逻辑为 BBOX 裁剪**
            # bbox_cropped_feature = crop_image_with_bbox(x, bboxes_points, target_size=_p3.shape[2:])
            #P3 = crop_image_with_centerpoint(x, bboxes_points, target_size=x3.shape[2:])
            #P3 = F.interpolate(P3, size=x3.shape[2:], mode='bilinear', align_corners=True)
            pure_image_patch = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
            _, lh3, hl3, hh3  = self.dwt(pure_image_patch)
            #P3_fused = self.wavelet_processor4.process_ll(ll_subband=ll_subband, inref_patch=P3)
            high_wavelet_3 = self.wavelet_processor4.process_high(lh3, hl3, hh3)


        px3_edge = self.edge_convs[0](F_next_3_refined)

        F_next_enc_proj = self.proj_e2(x2) 
        F_next_input_2, F_CRFB_3 = self.crfb3(F_next_3_refined, px3_edge, F_next_enc_proj)
        px3_edge = F.interpolate(px3_edge, size=F_next_input_2.shape[2:], mode='bilinear', align_corners=True)
        high_wavelet_3 = F.interpolate(high_wavelet_3, size=F_next_input_2.shape[2:], mode='bilinear', align_corners=True)
        detail_packet_3 = torch.cat([high_wavelet_3, px3_edge], dim=1)
        F_next_2_refined = self.lt_refiner3(F_next_input_2, detail_packet_3)
        M3_up = F.interpolate(self.aux_pred_heads[1](F_CRFB_3), size=(H, W), mode='bilinear', align_corners=True)
        seg_preds.append(M3_up)

        if self.training and self.config.out_ref:
            edge_pred_3 = self.edge_pred_heads[1](px3_edge)
            outs_edge_pred.append(F.interpolate(edge_pred_3, size=(H, W), mode='bilinear'))
            outs_edge_label.append(canny_gt * torch.sigmoid(M3_up.detach()))

        if self.config.dec_ipt:
            # **修改：替换 image2patches 逻辑为 BBOX 裁剪**
            # bbox_cropped_feature = crop_image_with_bbox(x, bboxes_points, target_size=_p2.shape[2:])
            #P2 = crop_image_with_centerpoint(x, bboxes_points, target_size=x2.shape[2:])
            #P2 = F.interpolate(P2, size=x2.shape[2:], mode='bilinear', align_corners=True)
            pure_image_patch = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
            _, lh2, hl2, hh2 = self.dwt(pure_image_patch)
            #t = self.ipt_blk3(P2)
            #P2_fused = self.wavelet_processor3.process_ll(ll_subband=ll_subband, inref_patch=P2)
            high_wavelet_2 = self.wavelet_processor3.process_high(lh2, hl2, hh2)
            #_p2 = torch.cat((_p2, t_fused), 1)
        
        px2_edge = self.edge_convs[1](F_next_2_refined)

        F_next_enc_proj = self.proj_e1(x1) 
        F_next_input_1, F_CRFB_2 = self.crfb2(F_next_2_refined, px2_edge, F_next_enc_proj)
        px2_edge = F.interpolate(px2_edge, size=F_next_input_1.shape[2:], mode='bilinear', align_corners=True)
        high_wavelet_2 = F.interpolate(high_wavelet_2, size=F_next_input_1.shape[2:], mode='bilinear', align_corners=True)
        detail_packet_2 = torch.cat([high_wavelet_2, px2_edge], dim=1)
        F_next_1_refined= self.lt_refiner2(F_next_input_1, detail_packet_2)
        M2_up = F.interpolate(self.aux_pred_heads[2](F_CRFB_2), size=x.shape[2:], mode='bilinear', align_corners=True)
        seg_preds.append(M2_up) # M2
        
        if self.training and self.config.out_ref:
            edge_pred_2 = self.edge_pred_heads[2](px2_edge)
            outs_edge_pred.append(F.interpolate(edge_pred_2, size=(H, W), mode='bilinear'))
            outs_edge_label.append(canny_gt * torch.sigmoid(M2_up.detach()))


        if self.config.dec_ipt:
            # **修改：替换 image2patches 逻辑为 BBOX 裁剪**
            # bbox_cropped_feature = crop_image_with_bbox(x, bboxes_points, target_size=_p1.shape[2:])
            #P1 = crop_image_with_centerpoint(x, bboxes_points, target_size=x1.shape[2:])
            #P1 = F.interpolate(P1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            pure_image_patch = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
            #_, _, _, _ = self.dwt(pure_image_patch)
            #t = self.ipt_blk2(P1)
            #P1_fused = self.wavelet_processor2.process_ll(ll_subband=ll_subband, inref_patch=P1)
            #_p1 = torch.cat((_p1, t_fused), 1)

        px1_edge = self.edge_convs[2](F_next_1_refined)

        F_next_enc_proj = self.proj_e0(x)
        F_final_feat, F_CRFB_1 = self.crfb1(F_next_1_refined, px1_edge, F_next_enc_proj)
        
    
        # 最终分割图 M1 (插值回原图尺寸)
        M1 = self.final_pred_head(F_final_feat)
        M1 = F.interpolate(M1, size=x.shape[2:], mode='bilinear', align_corners=True)        
        seg_preds.append(M1) # M1

        if self.training and self.config.out_ref:
            edge_pred_1 = self.edge_pred_heads[3](px1_edge)
            outs_edge_pred.append(F.interpolate(edge_pred_1, size=(H, W), mode='bilinear'))
            outs_edge_label.append(canny_gt * torch.sigmoid(M1.detach()))

        # --- C. 返回结果 (按 M4, M3, M2, M1 顺序) ---
        if self.training and self.config.out_ref:
            return [outs_edge_pred, outs_edge_label], seg_preds
        else:
            return seg_preds
        


