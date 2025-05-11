import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   

    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)

    return output



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x

class Mlp(nn.Module):
    """MLP with Dropout as used in Vision Transformer, MLP-Mixer, etc."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid()
        )
    def forward(self, e1, e2):
        concat = torch.cat([e1, e2], dim=-1)   # shape: (batch, 2*d)
        g = self.gate(concat)   # shape: (batch, d)
        h = torch.cat([g * e1, (1 - g) * e2], dim=1)              # element-wise
        return h                               # or return torch.cat([h, e1, e2], dim=-1)

class MultiGateFusion(nn.Module):
    def __init__(self, d_model, groups=4):
        super().__init__()
        self.groups = groups
        self.group_dim = d_model // groups
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.group_dim * 2, self.group_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.group_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, dwt_feat, clip_feat):
        fused = []
        for i in range(self.groups):
            dwt_part = dwt_feat[:, i*self.group_dim:(i+1)*self.group_dim]
            clip_part = clip_feat[:, i*self.group_dim:(i+1)*self.group_dim]
            fused_input = torch.cat([dwt_part, clip_part], dim=1)
            gate = self.gate_mlp(fused_input)
            # print(gate)
            fused_part = gate * dwt_part + (1 - gate) * clip_part
            fused.append(fused_part)
        return torch.cat(fused, dim=1)

import loralib as lora
import torch.nn as nn
def apply_lora_to_convnext(module, lora_rank=8):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # 检查 kernel_size 是否为 int
            k = child.kernel_size
            if isinstance(k, tuple):
                if k[0] != k[1]:
                    continue  # 跳过非正方卷积
                k = k[0]
            if not isinstance(k, int):
                continue

            try:
                new_layer = lora.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=k,
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias is not None,
                    r=lora_rank
                )
                # 用 state_dict 拷贝权重，避免属性访问问题
                new_layer.load_state_dict(child.state_dict())
                setattr(module, name, new_layer)
            except Exception as e:
                print(f"[LoRA warning] Failed to convert layer '{name}': {e}")
                continue

        else:
            apply_lora_to_convnext(child, lora_rank)
class AIDE_Model(nn.Module):
    def __init__(self, resnet_path, convnext_path, use_clip, use_resnet=True):
        super(AIDE_Model, self).__init__()
        self.use_clip = use_clip
        self.use_resnet = use_resnet

        if self.use_resnet:
            self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = Mlp(1024, 512, 2, act_layer=nn.GELU)
        # self.fc_clip = Mlp(256, 128, 2, act_layer=nn.GELU)
        # self.fc_res = nn.Linear(512, 2)
        # self.hpf = HPF()
        # self.gate_layer = MultiGateFusion(512, groups=1)
        # self.gate_layer = GatedFusion(512)
        if self.use_clip:
            print("build model with convnext_xxl")
            self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge", pretrained=convnext_path
            )

            self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
            self.openclip_convnext_xxl.head.global_pool = nn.Identity()
            self.openclip_convnext_xxl.head.flatten = nn.Identity()

            self.openclip_convnext_xxl.eval()
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.convnext_proj = nn.Sequential(
                nn.Linear(3072, 512),
            )
            apply_lora_to_convnext(self.openclip_convnext_xxl, lora_rank=8)

            for name, param in self.openclip_convnext_xxl.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False

            # for param in self.openclip_convnext_xxl.parameters():
            #     param.requires_grad = False

   

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        '''
        pip install pywavelets pytorch_wavelets
        '''
        from pytorch_wavelets import DWTForward, DWTInverse
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        Yl, Yh = DWT_filter(x)
        Yh_LH = Yh[0][:, :, 0, :, :]  # LH水平
        Yh_HL = Yh[0][:, :, 1, :, :]  # HL垂直
        Yh_HH = Yh[0][:, :, 2, :, :]  # HH对角
        Yh_LH = transforms.Resize([x.shape[-2], x.shape[-1]])(Yh_LH)
        Yh_HL = transforms.Resize([x.shape[-2], x.shape[-1]])(Yh_HL)
        Yh_HH = transforms.Resize([x.shape[-2], x.shape[-1]])(Yh_HH)
        
        return Yh_HH

    def split_to_patches(self, x, patch_size=32):
        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

        num_patches_h = H // patch_size  # = 8
        num_patches_w = W // patch_size  # = 8

        # unfold height & width
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B, C, 8, 8, 32, 32)

        # rearrange to (B, 64, C, 32, 32)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches_h * num_patches_w, C, patch_size, patch_size)

        return patches

    def forward(self, x):
        # x: [B, 3, 3, 256, 256]
        x_dwt = x[:, 0]
        x_clip = x[:, 1]
        B = x_dwt.shape[0]
        x_dwt_patch = self.split_to_patches(x_dwt)  # [B, 64, 3, 32, 32]
        # x_dwt = self._preprocess_dwt(x_dwt)
        # ============ ConvNeXt branch ============

        if self.use_clip:
            # with torch.no_grad():
            tokens = x_clip
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            feat_clip0 = self.convnext_proj(local_convnext_image_feats)
            # x_clip = self.fc_clip(feat_clip0)

        # ============ ResNet branch ============
        if self.use_resnet:
            # x_dwt_patch = x_dwt_patch.view(-1, 3, 32, 32)
            # patch_tokens = self.resnet(x_dwt_patch)
            # patch_tokens = patch_tokens.view(B, 64, -1)
            # # 输入 transformer 编码器
            # encoded = self.encoder(patch_tokens)             # (64, B, 512)
            # # 平均池化所有 patch 输出
            # feat_res0 = encoded[:, 0, :]                    # (B, 512)
            feat_res0 = self.resnet(x_dwt)


        # ============ Fusion and classification ============
        fused_feat  = torch.cat([feat_res0, feat_clip0], dim=1)  # [B, 1024]
        # fused_feat = self.gate_layer(feat_res0, feat_clip0)  # [B, 512]

        x = self.fc(fused_feat)  # [B, 2]
        # x = self.fc(feat_clip0)  # [B, 2]

        return x, x, x


def AIDE(resnet_path, convnext_path, use_clip=True, use_resnet=True):
    model = AIDE_Model(resnet_path, convnext_path, use_clip=use_clip, use_resnet=use_resnet)
    return model


