"""ShadowReward: a ranking-based reward model for shadow removal.

A frozen DINOv2 ViT-B/14 encodes the shadow input and a candidate (deshadowed) image into
patch tokens. The tokens of the two images are concatenated per patch and a small
convolutional head predicts a per-patch score in [0, 1]. The image-level reward is the mean
of the top-K patch scores (K = 10% of the patches).

The head is trained with the ListMLE ranking loss on synthetic ranked lists
(see dataloaders/shadow_dataset.py, ShadowRankingDataset). No human labels are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ConvTokenHead2D(nn.Module):
    def __init__(self, in_ch, mid=256, k=3, dilation=1, dropout=0.0):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.proj1 = nn.Conv2d(in_ch, mid, 1)
        self.dw = nn.Conv2d(mid, mid // 2, k, padding=pad, dilation=dilation, groups=mid // 2)
        self.pw = nn.Conv2d(mid // 2, mid // 4, 1)
        self.out = nn.Conv2d(mid // 4, 1, 1)
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.GELU()

    def forward(self, feat_tok, H, W):  # feat_tok: [B, N, F]
        B, N, Fd = feat_tok.shape
        x = feat_tok.view(B, H, W, Fd).permute(0, 3, 1, 2).contiguous()  # [B, F, H, W]
        x = self.act(self.proj1(x))
        x = self.act(self.dw(x))
        x = self.act(self.pw(x))
        x = self.drop(x)
        return torch.sigmoid(self.out(x)).squeeze(1)  # [B, H, W]


class ShadowReward(nn.Module):
    """Inputs are RGB images in [0, 1] with H and W multiples of 14 (we use 448 x 448)."""

    def __init__(self, device="cuda", feature_mode="concat", normalize_tokens=False, dropout=0.0, middle_dim=1024):
        super().__init__()
        self.device = torch.device(device)
        self.feature_mode = feature_mode
        self.normalize_tokens = normalize_tokens

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Frozen DINOv2 ViT-B/14 (downloaded through torch.hub)
        self.encoder = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14").to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.hidden_dim = getattr(self.encoder, "embed_dim", getattr(self.encoder, "width", 768))
        ps = getattr(self.encoder, "patch_size", None)
        if ps is None and hasattr(self.encoder, "patch_embed"):
            ps = getattr(self.encoder.patch_embed, "patch_size", 14)
        if isinstance(ps, tuple):
            ps = ps[0]
        self.patch = ps if ps is not None else 14

        self.per_tok_in = self.hidden_dim * (4 if self.feature_mode == "concat_diff" else 2)
        self.token_head = ConvTokenHead2D(in_ch=self.per_tok_in, mid=middle_dim, k=3, dilation=1,
                                          dropout=dropout).to(self.device)

    def _encode_patches(self, img: torch.Tensor) -> torch.Tensor:
        img = TF.normalize(img, mean=self.mean, std=self.std)
        toks = self.encoder.forward_features(img.to(self.device))["x_norm_patchtokens"]  # [B, N, D]
        if self.normalize_tokens:
            toks = F.normalize(toks, dim=-1)
        return toks

    def _pair_features(self, s_tok, c_tok):
        if self.feature_mode == "concat":
            return torch.cat([s_tok, c_tok], dim=-1)
        diff = s_tok - c_tok
        return torch.cat([s_tok, c_tok, diff, diff.abs()], dim=-1)

    def set_train(self):
        self.encoder.eval()
        self.token_head.train()
        for p in self.token_head.parameters():
            p.requires_grad = True

    @staticmethod
    def _reduce(flat, reduce, topk):
        """flat: [..., N] patch scores."""
        if reduce is None:
            return flat
        if reduce in ["avg", "mean"]:
            return flat.mean(dim=-1)
        if reduce == "max":
            return flat.max(dim=-1).values
        if reduce == "topk":
            N = flat.size(-1)
            if isinstance(topk, float):
                assert 0 < topk <= 1, "a float topk must be a ratio in (0, 1]"
                k = max(1, int(round(N * topk)))
            elif isinstance(topk, int):
                k = min(max(1, topk), N)
            else:
                raise ValueError("topk must be a float (ratio) or an int (count)")
            return flat.topk(k, dim=-1).values.mean(dim=-1)
        raise ValueError("reduce must be one of: None | avg | mean | max | topk")

    def forward(self, shadow_img, cand_img, reduce="topk", topk=0.1):
        """Returns the reward [B] (or the score map [B, H/14, W/14] if reduce is None)."""
        B, _, H, W = shadow_img.shape
        assert H % self.patch == 0 and W % self.patch == 0, f"Input size must be multiple of patch={self.patch}"
        Ht, Wt = H // self.patch, W // self.patch

        s_tok = self._encode_patches(shadow_img)
        c_tok = self._encode_patches(cand_img)
        feat = self._pair_features(s_tok, c_tok)
        scoremap = self.token_head(feat, Ht, Wt)  # [B, Ht, Wt]

        if reduce is None:
            return scoremap
        return self._reduce(scoremap.view(B, -1), reduce, topk)

    def score_candidates(self, shadow, candidates, reduce="topk", topk=0.1):
        """shadow: [B, 3, H, W], candidates: [B, K, 3, H, W] -> scores [B, K]."""
        B, K = candidates.shape[:2]
        _, _, H, W = shadow.shape
        assert H % self.patch == 0 and W % self.patch == 0, f"Input size must be multiple of patch={self.patch}"
        Ht, Wt = H // self.patch, W // self.patch
        N = Ht * Wt

        s_tok = self._encode_patches(shadow)  # [B, N, D]
        c_flat = candidates.view(B * K, *candidates.shape[2:])
        c_tok = self._encode_patches(c_flat)  # [B*K, N, D]
        s_tok_rep = s_tok.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(B * K, s_tok.size(1), s_tok.size(2))

        feat_tok = self._pair_features(s_tok_rep, c_tok)
        tok_map = self.token_head(feat_tok, Ht, Wt)  # [B*K, Ht, Wt]
        if reduce is None:
            return tok_map.view(B, K, Ht, Wt)
        return self._reduce(tok_map.view(B, K, N), reduce, topk)

    @staticmethod
    def listmle_loss(scores, ranks, temperature=1.0, jitter=1e-3):
        """ListMLE loss. scores: [B, K] (higher is better), ranks: [B, K] (0 is the best)."""
        B, K = scores.shape
        ranks_f = ranks.to(dtype=scores.dtype)
        noisy = ranks_f + jitter * torch.rand_like(ranks_f)
        perm = torch.argsort(noisy, dim=1)
        s_sorted = torch.gather(scores / temperature, 1, perm)

        loss = 0.0
        for i in range(K):
            denom = torch.logsumexp(s_sorted[:, i:], dim=1)
            numer = s_sorted[:, i]
            loss = loss + (denom - numer)
        return loss.mean()

    def get_listwise_loss(self, shadow, candidates, ranks, reduce="topk", topk=0.1, temperature=1.0, jitter=1e-3):
        scores = self.score_candidates(shadow, candidates, reduce=reduce, topk=topk)
        return self.listmle_loss(scores, ranks, temperature=temperature, jitter=jitter)

    def save(self, path: str):
        sd = {}
        sd["token_head"] = self.token_head.state_dict()
        sd["hidden_dim"] = self.hidden_dim
        sd["feature_mode"] = self.feature_mode
        sd["normalize_tokens"] = self.normalize_tokens
        torch.save(sd, path)
        print(f"[ShadowReward] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.token_head.load_state_dict(ckpt["token_head"], strict=True)
        print(f"[ShadowReward] Loaded from {path}")
