# --------------------------------------------------------
# This code is borrowed from https://github.com/MishaLaskin/curl
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import random
import torchvision.transforms.functional as TF
from vit_modules import *
from utils import PositionEmbeddingRandom
from tools import mask_img

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

def color_jitter_(img, perm, b, c, s, h):
        img1 = img[:, 0:3]
        img2 = img[:, 3:6]
        img3 = img[:, 6:9]

        for fn_id in perm:
            if fn_id == 0:
                img1 = TF.adjust_brightness(img1, b)
                img2 = TF.adjust_brightness(img2, b)
                img3 = TF.adjust_brightness(img3, b)
            elif fn_id == 1:
                img1 = TF.adjust_contrast(img1, c)
                img2 = TF.adjust_contrast(img2, c)
                img3 = TF.adjust_contrast(img3, c)
            elif fn_id == 2:
                img1 = TF.adjust_saturation(img1, s)
                img2 = TF.adjust_saturation(img2, s)
                img3 = TF.adjust_saturation(img3, s)
            elif fn_id == 3:
                img1 = TF.adjust_hue(img1, h)
                img2 = TF.adjust_hue(img2, h)
                img3 = TF.adjust_hue(img3, h)

        img = torch.cat((img1, img2, img3), 1)

        return img

def view_as_windows_cuda(x, window_shape):
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)

def add_random_boxes_cuda(x, boxes=15, mu=10, sigma=10, size_min=5, size_max=28, confuse_color=None, confuse_prob=0.0):
	n = x.shape[0]
	img_size = x.shape[-1]

	x_perm = x.permute(0, 2, 3, 1)

	boxes = int(boxes)

	for _ in range(boxes):
		size = int(random.gauss(mu=mu, sigma=sigma))
		if size < size_min:
			size = size_min
		if size > size_max:
			size = size_max
		crop_max = img_size - size
		w1 = torch.LongTensor(n).random_(0, crop_max)
		h1 = torch.LongTensor(n).random_(0, crop_max)

		windows = view_as_windows_cuda(x_perm, (1, size, size, 1))[..., 0, :, :, 0]
		# n, 3, size, size
		if confuse_prob and random.random() < confuse_prob:  # confuse_color: n, 3, 1, 1
			windows[torch.arange(n), w1, h1] = confuse_color + torch.randn_like(confuse_color) * 0.2
		else:
			windows[torch.arange(n), w1, h1] = torch.rand((n, 9, 1, 1), device=windows.device)*255

	return x

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self,
                 obs_shape,
                 feature_dim,
                 num_layers=2,
                 num_filters=32,
                 output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[
            num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, add_aug, add_mask_img, spatial, position_Embedding, flatten=True):
        dg = 1
        if add_aug:
            random_aug1 = random.randint(0, 1)
            if random_aug1:
                zeros1 = torch.zeros_like(obs[0::2, :, :, :])
                random_boxes1 = add_random_boxes_cuda(zeros1, boxes=50, mu=0, sigma=2, size_min=1, size_max=800)
                obs[0::2, :, :, :] += random_boxes1
            else:
                perm1 = torch.randperm(4)
                b_1 = random.gauss(1, 0.5 * dg)
                c_1 = random.gauss(1, 0.5 * dg)
                s_1 = random.gauss(1, 0.5 * dg)
                h_1 = random.gauss(0, 0.4 * dg)
                b_1 = np.clip(b_1, 0.5, 2)
                c_1 = np.clip(c_1, 0.5, 2)
                s_1 = np.clip(s_1, 0.5, 5)
                h_1 = np.clip(h_1, 0, 0.5)
                obs[0::2] = color_jitter_(obs[0::2] / 255, perm1, b_1, c_1, s_1, h_1) * 255

            random_aug2 = random.randint(0, 1)

            if random_aug2:
                zeros2 = torch.zeros_like(obs[1::2, :, :, :])
                random_boxes2 = add_random_boxes_cuda(zeros2, boxes=50, mu=0, sigma=2, size_min=1, size_max=800)
                obs[1::2, :, :, :] += random_boxes2
            else:
                perm2 = torch.randperm(4)
                b_2 = random.gauss(1, 0.5 * dg)
                c_2 = random.gauss(1, 0.5 * dg)
                s_2 = random.gauss(1, 0.5 * dg)
                h_2 = random.gauss(0, 0.4 * dg)
                b_2 = np.clip(b_2, 0.5, 2)
                c_2 = np.clip(c_2, 0.5, 2)
                s_2 = np.clip(s_2, 0.5, 5)
                h_2 = np.clip(h_2, 0, 0.5)
                obs[1::2] = color_jitter_(obs[1::2] / 255, perm2, b_2, c_2, s_2, h_2) * 255

        if spatial:
            obs = torch.add(obs, position_Embedding)

        if add_mask_img:
            obs = mask_img(obs)

        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1) if flatten else conv        
        return h

    def forward(self, obs, detach=False, add_mask_img=False, add_aug=False, spatial=False, position_Embedding=None):
        h = self.forward_conv(obs, add_aug, add_mask_img, spatial, position_Embedding)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass



class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    # def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
    #              num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
    #              drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
    #              use_learnable_pos_emb=False):
    def __init__(self, img_size=84, patch_size=7, in_chans=9, num_classes=0, embed_dim=441, depth=4,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0,
                 use_learnable_pos_emb=False, feature_dim=50):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # embed_dim = 128
        # self.conv = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # num_patches += 1

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # self.state = nn.Sequential(
        #     nn.Linear(embed_dim, feature_dim),
        #     nn.LayerNorm(feature_dim)
        # )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x) # (B, NumPatches*NumPatches, C=9*7*7)
        
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        # x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        x_vis = x if mask is None else x[:, ~mask].reshape(B, -1, C)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        # x = x.mean(1)
        # # x = x[:, 0]
        # x = x.detach() if detach else x
        # x = self.state(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=84, patch_size=12, in_chans=9, num_classes=0, embed_dim=512, depth=4,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0,
                 use_learnable_pos_emb=False, feature_dim=50):
        super().__init__()
        self.feature_dim = feature_dim

        self.vit = PretrainVisionTransformerEncoder(
            img_size, patch_size, in_chans, num_classes, embed_dim, depth, 
            num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, 
            drop_path_rate, norm_layer, init_values, use_learnable_pos_emb, feature_dim
        )


        self.state = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, x, mask=None, detach=False):
        x = x / 255.
        x = self.vit(x, mask)
        x = x.mean(1)
        x = x.detach() if detach else x
        # x = self.state(x)
        return x

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        # for (src_child, tgt_child) in zip(source.vit.children(), self.vit.children()):
        for (src_module, tgt_module) in zip(source.vit.modules(), self.vit.modules()):
            if isinstance(src_module, nn.Module):
                try:
                    # print("Tie: ", src_module)
                    tie_weights(src=src_module, trg=tgt_module)
                except:
                    # print("Skip: ", src_module)
                    pass
        # for i in range(self.num_layers):
        #     tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        pass

_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder,
    # 'pixel': ViTEncoder,
    'identity': IdentityEncoder
}

def make_encoder(encoder_type,
                 obs_shape,
                 feature_dim,
                 num_layers,
                 num_filters,
                 output_logits=False):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim,
                                             num_layers, num_filters,
                                             output_logits)
    # return _AVAILABLE_ENCODERS[encoder_type](
    #     img_size=obs_shape[1], 
    #     patch_size=8, 
    #     embed_dim=128, 
    #     depth=4,
    #     num_heads=8,
    #     feature_dim=feature_dim)