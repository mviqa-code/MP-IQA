import json

import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

from .clip import clip


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_features):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_features.argmax(dim=-1)] @ self.text_projection
        return x


class ImageEncoder(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.transformer = visual.transformer
        self.ln_pre = visual.ln_pre
        self.ln_post = visual.ln_post
        self.proj = visual.proj

    def forward(self, x, p):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        if p is not None:
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), p, x], dim=1)
        else:
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj
        return x


class MultiModalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        bunch_layer = nn.TransformerDecoderLayer(
            d_model=config.MODEL.DIM,
            dropout=config.MODEL.DROPOUT,
            nhead=config.MODEL.MMD_NHEAD,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(config.MODEL.DIM * 4),
            norm_first=True
        )
        self.bunch_decoder = nn.TransformerDecoder(bunch_layer, num_layers=config.MODEL.MMD_NUM_LAYERS)
        self.ln = nn.LayerNorm(config.MODEL.DIM)
        self.mlp = nn.Sequential(
            nn.Linear(config.MODEL.DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(config.MODEL.DROPOUT),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(config.MODEL.DROPOUT),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, q, k_v):
        decoded_features = self.bunch_decoder(q, k_v)
        decoded_features = self.ln(decoded_features)
        decoded_features = self.mlp(decoded_features).squeeze(-1)
        score = torch.mean(decoded_features, dim=1)
        return score


class ImageDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.MODEL.DIM)
        self.mlp = nn.Sequential(
            nn.Linear(config.MODEL.DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(config.MODEL.DROPOUT),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(config.MODEL.DROPOUT),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.ln(x)
        decoded_features = self.mlp(x).squeeze(-1)
        score = torch.mean(decoded_features, dim=1)
        return score


class TextualPromptLearner(nn.Module):
    def __init__(self, config, dtype, token_embedding):
        super().__init__()
        categories = config.categories
        num_category = len(categories)

        n_ctx = config.MODEL.N_CTX_C
        ctx_dim = config.MODEL.CTX_DIM
        ctx_vectors = torch.empty(num_category, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = ' '.join(['X'] * n_ctx)

        prompts = [prompt_prefix + ' ' + name + '.' for name in categories]
        tokenized_features = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_features).type(dtype)

        self.ctx = nn.Parameter(ctx_vectors)
        self.tokenized_features = tokenized_features
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])  # Category + EOS + PAD

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix
        tokenized_features = self.tokenized_features
        prompts = torch.cat([
            prefix,
            ctx,
            suffix
        ], dim=1)
        return prompts, tokenized_features


class VisualPromptLearner(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        dim = config.MODEL.DIM
        vision_width = config.MODEL.VISION_WIDTH
        scale = vision_width ** -0.5
        self.prompt_vectors = nn.Parameter(scale * torch.randn(1, config.MODEL.N_CTX_L, dim, dtype=dtype))
        self.prompt_proj = nn.Linear(dim, vision_width)
        self.prompt_dropout = nn.Dropout(config.MODEL.DROPOUT)

    def forward(self):
        prompt = self.prompt_vectors
        prompt = self.prompt_proj(prompt)
        prompt = self.prompt_dropout(prompt)
        return prompt


class MPIQA(nn.Module):
    def __init__(self, config, clip_model):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.image_decoder = ImageDecoder(config)
        self.multimodal_decoder = MultiModalDecoder(config)
        self.textual_prompt_learner = TextualPromptLearner(config, clip_model.dtype, clip_model.token_embedding)
        self.visual_prompt_learner = VisualPromptLearner(config, clip_model.dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.textual = config.textual
        self.visual = config.visual
        self.num_ctx_l = config.MODEL.N_CTX_L

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    def forward(self, x, train_mode=False):
        if train_mode:
            if self.textual and self.visual:
                B = x.shape[0]
                tp, tokenized_features = self.textual_prompt_learner()
                textual_prompt = self.text_encoder(tp, tokenized_features).unsqueeze(0).expand(B, -1, -1)
                textual_prompt = textual_prompt / textual_prompt.norm(dim=-1, keepdim=True)

                vp = self.visual_prompt_learner()
                x = self.image_encoder(x, vp)
                cls_feature, visual_prompt, patch_feature = x[:, :1, :], x[:, 1:self.num_ctx_l+1, :], x[:, self.num_ctx_l+1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                visual_prompt = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits_category = logit_scale * torch.matmul(textual_prompt, cls_feature.permute(0, 2, 1)).squeeze(2)
                logits_location = logit_scale * torch.matmul(visual_prompt, patch_feature.permute(0, 2, 1))
                logits_location = logits_location.mean(dim=1)

                query = torch.cat([textual_prompt, visual_prompt], dim=1)
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score, logits_category, logits_location

            elif self.visual:
                vp = self.visual_prompt_learner()
                x = self.image_encoder(x, vp)
                cls_feature, visual_prompt, patch_feature = x[:, :1, :], x[:, 1:self.num_ctx_l+1, :], x[:, self.num_ctx_l+1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                visual_prompt = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits_location = logit_scale * torch.matmul(visual_prompt, patch_feature.permute(0, 2, 1))
                logits_location = logits_location.mean(dim=1)

                query = visual_prompt
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score, None, logits_location

            elif self.textual:
                B = x.shape[0]
                tp, tokenized_features = self.textual_prompt_learner()
                textual_prompt = self.text_encoder(tp, tokenized_features).unsqueeze(0).expand(B, -1, -1)
                textual_prompt = textual_prompt / textual_prompt.norm(dim=-1, keepdim=True)

                x = self.image_encoder(x, None)
                cls_feature, patch_feature = x[:, :1, :], x[:, 1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits_category = logit_scale * torch.matmul(textual_prompt, cls_feature.permute(0, 2, 1)).squeeze(2)

                query = textual_prompt
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score, logits_category, None

            else:
                x = self.image_encoder(x, None)
                predict_score = self.image_decoder(x)

                return predict_score, None, None

        else:
            if self.textual and self.visual:
                B = x.shape[0]
                tp, tokenized_features = self.textual_prompt_learner()
                textual_prompt = self.text_encoder(tp, tokenized_features).unsqueeze(0).expand(B, -1, -1)
                textual_prompt = textual_prompt / textual_prompt.norm(dim=-1, keepdim=True)

                vp = self.visual_prompt_learner()
                x = self.image_encoder(x, vp)
                cls_feature, visual_prompt, patch_feature = x[:, :1, :], x[:, 1:self.num_ctx_l+1, :], x[:, self.num_ctx_l+1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                visual_prompt = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                query = torch.cat([textual_prompt, visual_prompt], dim=1)
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score

            elif self.visual:
                vp = self.visual_prompt_learner()
                x = self.image_encoder(x, vp)
                cls_feature, visual_prompt, patch_feature = x[:, :1, :], x[:, 1:self.num_ctx_l+1, :], x[:, self.num_ctx_l+1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                visual_prompt = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                query = visual_prompt
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score

            elif self.textual:
                B = x.shape[0]
                tp, tokenized_features = self.textual_prompt_learner()
                textual_prompt = self.text_encoder(tp, tokenized_features).unsqueeze(0).expand(B, -1, -1)
                textual_prompt = textual_prompt / textual_prompt.norm(dim=-1, keepdim=True)

                x = self.image_encoder(x, None)
                cls_feature, patch_feature = x[:, :1, :], x[:, 1:, :]
                cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                query = textual_prompt
                key_value = torch.cat([cls_feature, patch_feature], dim=1)
                predict_score = self.multimodal_decoder(query, key_value)

                return predict_score

            else:
                x = self.image_encoder(x, None)
                predict_score = self.image_decoder(x)

                return predict_score