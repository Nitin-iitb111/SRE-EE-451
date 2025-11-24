# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

import timm


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.activation(self.dropout(self.network(x)))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import clip  # make sure clip is installed: pip install git+https://github.com/openai/CLIP.git
# from PIL import Image
# from typing import List, Optional, Union, Iterable


# class CLIPWrapper(nn.Module):
#     """
#     Robust CLIP wrapper for DAMP integration.

#     Key features:
#       - load CLIP model and preprocess
#       - encode images (tensor or PIL) with batching and tqdm
#       - encode text with batching, tokenization caching and support for multiple templates
#       - simple prompt injection (embedding-level additive prompt by default)
#       - device/dtype handling and optional freezing of visual/text encoders
#       - small, clear API: set_classnames, set_templates, encode_image*, encode_text*
#     """

#     def __init__(
#         self,
#         model_name: str = "ViT-B/32",
#         device: Optional[str] = None,
#         freeze_visual: bool = True,
#         freeze_text: bool = False,
#         classnames: Optional[List[str]] = None,
#         templates: Optional[List[str]] = None,
#         verbose: bool = False,
#     ):

#         super(CLIPWrapper, self).__init__()

#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = device
#         self.verbose = verbose

#         if self.verbose:
#             print(f"[CLIPWrapper] Loading CLIP model '{model_name}' on {self.device} ...")

#         # load model + preprocess
#         self.model, self.preprocess = clip.load(model_name, device=self.device)
#         self.model.to(self.device)
#         # default to eval mode; user can fine-tune by unfreezing & calling train()
#         self.model.eval()

#         # dims (robust access)
#         try:
#             self.visual_embedding_dim = (
#                 self.model.visual.output_dim if hasattr(self.model, "visual") else self.model.visual_embed_dim
#             )
#         except Exception:
#             # fallback
#             self.visual_embedding_dim = getattr(self.model, "visual_embed_dim", None)

#         try:
#             self.text_embedding_dim = (
#                 self.model.transformer.width if hasattr(self.model, "transformer") else None
#             )
#         except Exception:
#             self.text_embedding_dim = None

#         # fallback: try to get embed by quick forward (cheap)
#         if self.text_embedding_dim is None:
#             # create a dummy token and forward once (fast)
#             try:
#                 tok = clip.tokenize(["hello"]).to(self.device)
#                 with torch.no_grad():
#                     t_emb = self.model.encode_text(tok)
#                 self.text_embedding_dim = t_emb.shape[-1]
#             except Exception:
#                 self.text_embedding_dim = 512  # reasonable default

#         # classnames/templates
#         self.classnames = list(classnames) if classnames is not None else []
#         # default template if none provided
#         self.templates = list(templates) if templates is not None else ["a photo of a {}"]

#         # logit scale (float) from CLIP model (store scalar, not Parameter here)
#         try:
#             self.logit_scale = float(self.model.logit_scale.item())
#         except Exception:
#             self.logit_scale = 1.0

#         # freeze selective parts if requested
#         if freeze_visual:
#             try:
#                 for p in self.model.visual.parameters():
#                     p.requires_grad = False
#             except Exception:
#                 # some CLIP variants might differ; ignore if not present
#                 pass
#         if freeze_text:
#             try:
#                 for p in self.model.transformer.parameters():
#                     p.requires_grad = False
#             except Exception:
#                 pass

#         # tokenization cache: map from tuple(texts, templates) -> tokenized tensor (on device)
#         self._token_cache = {}
#         # cached base class embeddings (no prompt added) - (C, D_text)
#         self._cached_base_class_embeds = None
#         self._cached_base_class_embeds_for_templates = None  # tracks templates used

#         if self.verbose:
#             print("[CLIPWrapper] CLIP model loaded successfully ✅")
#             print(f"[CLIPWrapper] visual_dim={self.visual_embedding_dim}, text_dim={self.text_embedding_dim}")
#             if len(self.classnames) > 0:
#                 print(f"[CLIPWrapper] {len(self.classnames)} classnames configured.")

#     # ---------- Utilities ----------
#     def set_classnames(self, classnames: Iterable[str], clear_cache: bool = True):
#         """Set or update classnames used for class text encodings."""
#         self.classnames = list(map(str, classnames))
#         if clear_cache:
#             self.clear_token_cache()

#     def set_templates(self, templates: Iterable[str], clear_cache: bool = True):
#         """Set templates like 'a photo of a {}' or multiple templates."""
#         self.templates = list(templates)
#         if clear_cache:
#             self.clear_token_cache()

#     def clear_token_cache(self):
#         """Clear tokenization / cached embeddings."""
#         self._token_cache.clear()
#         self._cached_base_class_embeds = None
#         self._cached_base_class_embeds_for_templates = None

#     def _prepare_texts_from_classnames(self, classnames: List[str]) -> List[str]:
#         """Apply templates to classnames and return list of texts.
#         If multiple templates provided, we use the average embedding across templates (per class)."""
#         texts_per_template = []
#         for template in self.templates:
#             texts = [template.format(c) for c in classnames]
#             texts_per_template.append(texts)
#         # If there are multiple templates, we will tokenize each template batch separately,
#         # encode them, and average embeddings later.
#         return texts_per_template

#     def _tokenize_and_cache(self, texts: List[str]):
#         """
#         Tokenize list of strings and cache the tokenized tensor on device.
#         Returns tokenized tensor on self.device.
#         """
#         key = tuple(texts)
#         if key in self._token_cache:
#             return self._token_cache[key]
#         # tokenize (clip.tokenize handles batching internally)
#         tok = clip.tokenize(list(texts)).to(self.device)
#         self._token_cache[key] = tok
#         return tok

#     def _batched_forward(self, fn, inputs: torch.Tensor, batch_size: int = 32, desc: Optional[str] = None):
#         """
#         Generic helper to forward inputs in batches through function `fn` which accepts a tensor.
#         fn should return a tensor for each sub-batch.
#         inputs: tensor of shape (N, ...) on CPU or GPU (we move sub-batches to self.device inside)
#         """
#         N = inputs.shape[0]
#         outs = []
#         if desc is None:
#             desc = "CLIP batched forward"
#         for i in range(0, N, batch_size):
#             sub = inputs[i : i + batch_size].to(self.device)
#             with torch.no_grad():
#                 out = fn(sub)
#             outs.append(out.detach().cpu())
#         if len(outs) == 0:
#             return torch.empty((0, fn(torch.zeros(1, device=self.device)).shape[-1]), dtype=torch.float)
#         return torch.cat(outs, dim=0).to(self.device).float()

#     # ---------- Image encoding ----------
#     def encode_image(self, images: Union[torch.Tensor, Iterable[Image.Image]], batch_size: int = 32):
#         """
#         Encode images to CLIP image embeddings.

#         Args:
#             images: either a torch.Tensor of shape (B, C, H, W) in range expected by CLIP,
#                     or an iterable of PIL images (will be preprocessed).
#             batch_size: mini-batch size for CLIP forward passes.

#         Returns:
#             Tensor of shape (B, D_visual) on self.device (float).
#         """
#         # If PIL images provided, preprocess them to tensors first
#         if not isinstance(images, torch.Tensor):
#             # assume iterable of PIL images
#             imgs = []
#             for im in images:
#                 imgs.append(self.preprocess(im))
#             images = torch.stack(imgs, dim=0)

#         # ensure dtype and device-handling
#         if images.dtype != torch.float32:
#             images = images.float()
#         N = images.shape[0]
#         if N == 0:
#             return torch.empty((0, self.visual_embedding_dim), device=self.device, dtype=torch.float)

#         # Use batched forward, but call model.encode_image for each batch
#         def fn(subbatch):
#             return self.model.encode_image(subbatch)

#         desc = f"[CLIPWrapper] Encoding images ({N} imgs)"
#         return self._batched_forward(fn, images.cpu(), batch_size=batch_size, desc=desc)

#     def encode_image_with_prompt(
#         self,
#         images: Union[torch.Tensor, Iterable[Image.Image]],
#         visual_prompt: Optional[torch.Tensor] = None,
#         batch_size: int = 32,
#         prompt_mode: str = "add_mean",
#     ):
#         """
#         Encode images then inject a visual prompt.

#         visual_prompt: Tensor of shape (L_vis, D_vis) or None.
#         prompt_mode:
#           - "add_mean": compute mean over prompt tokens and add to image embedding (default).
#           - "concat": Not implemented here (requires architectural change).
#         """
#         img_emb = self.encode_image(images, batch_size=batch_size)  # (B, D_vis)
#         if visual_prompt is None:
#             return img_emb
#         # ensure prompt on proper device/dtype
#         prompt = visual_prompt.to(self.device).float()
#         if prompt.ndim == 2:
#             prompt_avg = prompt.mean(dim=0, keepdim=True)  # (1, D)
#         else:
#             raise ValueError("visual_prompt must be 2D (L_vis, D_vis)")

#         if prompt_avg.shape[-1] != img_emb.shape[-1]:
#             # try to broadcast or raise
#             if prompt_avg.shape[-1] == self.visual_embedding_dim:
#                 # ok
#                 pass
#             else:
#                 raise ValueError(f"visual_prompt dim {prompt_avg.shape[-1]} != image embedding dim {img_emb.shape[-1]}")

#         # add prompt_avg to each image embedding
#         return img_emb + prompt_avg

#     # ---------- Text encoding ----------
#     def encode_text(self, texts: List[str], batch_size: int = 32, use_tqdm: bool = True):
#         """
#         Encode arbitrary list of text strings (not necessarily classnames).
#         Returns tensor (len(texts), D_text) on self.device.
#         """
#         if len(texts) == 0:
#             return torch.empty((0, self.text_embedding_dim), device=self.device, dtype=torch.float)

#         tokenized = self._tokenize_and_cache(texts)  # tokenized tensor on device
#         # Use batched forward over tokenized inputs
#         def fn(subtok):
#             return self.model.encode_text(subtok)

#         desc = f"[CLIPWrapper] Encoding text ({len(texts)} items)"
#         # tokenized is on device, but our _batched_forward expects CPU tensor to move to device per-batch.
#         # Move tokenized to cpu first to avoid device mismatch handling.
#         return self._batched_forward(fn, tokenized.cpu(), batch_size=batch_size, desc=desc)

#     def _compute_and_cache_base_class_embeds(self, batch_size: int = 32):
#         """
#         Compute and cache base class embeddings (for self.classnames) using the current templates,
#         WITHOUT adding any learnable prompt. Results stored in self._cached_base_class_embeds.
#         If multiple templates are used, we average embeddings across templates.
#         """
#         if len(self.classnames) == 0:
#             raise ValueError("CLIPWrapper: classnames is empty. Use set_classnames(...) before calling this.")

#         # If templates unchanged and cache exists, reuse
#         templates_key = tuple(self.templates)
#         if (self._cached_base_class_embeds is not None and
#                 self._cached_base_class_embeds_for_templates == templates_key):
#             return  # already computed

#         # for each template, build texts and encode
#         all_template_embeds = []
#         for template in self.templates:
#             texts = [template.format(c) for c in self.classnames]
#             emb = self.encode_text(texts, batch_size=batch_size)
#             all_template_embeds.append(emb)  # (C, D)

#         # average over templates (axis=0)
#         if len(all_template_embeds) == 1:
#             base_embeds = all_template_embeds[0]
#         else:
#             stacked = torch.stack(all_template_embeds, dim=0)  # (T, C, D)
#             base_embeds = stacked.mean(dim=0)  # (C, D)

#         self._cached_base_class_embeds = base_embeds.to(self.device).float()
#         self._cached_base_class_embeds_for_templates = templates_key
#         return

#     def encode_text_with_prompt(self, classnames: Optional[List[str]] = None, text_prompt: Optional[torch.Tensor] = None, batch_size: int = 32):
#         """
#         Return text embeddings for classes with added prompt embedding.
#         - classnames: optional list; if None uses self.classnames
#         - text_prompt: optional tensor (L_txt, D_txt) that will be averaged and added
#         """
#         if classnames is None:
#             classnames = self.classnames
#         else:
#             classnames = list(map(str, classnames))

#         # ensure cached base class embedding exists
#         # this uses the wrapper's templates and caches across calls
#         if (self._cached_base_class_embeds is None) or (self._cached_base_class_embeds_for_templates != tuple(self.templates)):
#             self._compute_and_cache_base_class_embeds(batch_size=batch_size)

#         base = self._cached_base_class_embeds  # (C, D)
#         if base is None or base.shape[0] == 0:
#             raise RuntimeError("CLIPWrapper: cached base class embeddings empty after computation.")

#         # add prompt (average over prompt tokens) if provided
#         if text_prompt is None:
#             return base.to(self.device).float()
#         prompt = text_prompt.to(self.device).float()
#         if prompt.ndim != 2:
#             raise ValueError("text_prompt must be 2D (L_txt, D_txt)")
#         prompt_avg = prompt.mean(dim=0, keepdim=True)  # (1, D)
#         if prompt_avg.shape[-1] != base.shape[-1]:
#             raise ValueError(f"Prompt dim {prompt_avg.shape[-1]} != text embed dim {base.shape[-1]}")

#         return (base + prompt_avg).to(self.device).float()

#     # ---------- small helper properties ----------
#     @property
#     def device_str(self):
#         return self.device

#     # To allow non-harsh attribute access outside this wrapper
#     @property
#     def dtype(self):
#         return torch.get_default_dtype()

# def get_class_names_from_dataset(dataset_obj):
#     """
#     Robustly extract class names from many Dataset wrappers used in DomainBed.
#     Returns list of class name strings or falls back to numeric string names
#     when only a class count is available.
#     """
#     # direct attribute used by some DomainBed/WILDS datasets
#     if hasattr(dataset_obj, "class_names") and dataset_obj.class_names is not None:
#         return list(dataset_obj.class_names)

#     # torchvision ImageFolder / some torchvision datasets
#     if hasattr(dataset_obj, "classes") and dataset_obj.classes is not None:
#         return list(dataset_obj.classes)

#     # DomainBed style: dataset object might expose num_classes
#     for attr in ("num_classes", "n_classes", "N_CLASSES", "n_labels"):
#         if hasattr(dataset_obj, attr):
#             n = int(getattr(dataset_obj, attr))
#             return [str(i) for i in range(n)]

#     # unwrap common wrappers (Subset, ConcatDataset, wrappers with .dataset/.datasets)
#     if hasattr(dataset_obj, "dataset") and dataset_obj.dataset is not dataset_obj:
#         return get_class_names_from_dataset(dataset_obj.dataset)
#     if hasattr(dataset_obj, "datasets"):
#         for d in dataset_obj.datasets:
#             try:
#                 names = get_class_names_from_dataset(d)
#                 if names:
#                     return names
#             except Exception:
#                 continue

#     # last resort: try to infer number of classes from targets/labels
#     try:
#         if hasattr(dataset_obj, "targets") and dataset_obj.targets is not None:
#             labels = dataset_obj.targets
#             n = int(max(labels)) + 1
#             return [str(i) for i in range(n)]
#     except Exception:
#         pass

#     raise ValueError("Dataset object has no attribute 'class_names' and class count could not be inferred.")

import torch
import torch.nn as nn
from tqdm import tqdm
import clip  # make sure clip is installed: pip install git+https://github.com/openai/CLIP.git
from PIL import Image
from typing import List, Optional, Union, Iterable


class CLIPWrapper(nn.Module):
    """
    Robust CLIP wrapper for DAMP integration.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        freeze_visual: bool = True,
        freeze_text: bool = False,
        classnames: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        super(CLIPWrapper, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.verbose = verbose

        if self.verbose:
            print(f"[CLIPWrapper] Loading CLIP model '{model_name}' on {self.device} ...")

        # load model + preprocess
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.to(self.device)
        self.model.eval()  # ✅ ensure eval mode by default

        # determine embedding dims robustly
        try:
            self.visual_embedding_dim = getattr(self.model.visual, "output_dim", None)
            if self.visual_embedding_dim is None:
                self.visual_embedding_dim = getattr(self.model, "visual_embed_dim", 512)
        except Exception:
            self.visual_embedding_dim = 512  # ✅ fallback default

        try:
            self.text_embedding_dim = getattr(self.model.transformer, "width", None)
        except Exception:
            self.text_embedding_dim = None

        # ✅ if text dim is unknown, quickly probe it
        if self.text_embedding_dim is None:
            try:
                tok = clip.tokenize(["hello"]).to(self.device)
                with torch.no_grad():
                    t_emb = self.model.encode_text(tok)
                self.text_embedding_dim = t_emb.shape[-1]
            except Exception:
                self.text_embedding_dim = 512

        # ✅ keep consistent types
        self.classnames = list(classnames) if classnames is not None else []
        self.templates = list(templates) if templates is not None else ["a photo of a {}"]

        # logit scale scalar
        try:
            self.logit_scale = float(self.model.logit_scale.item())
        except Exception:
            self.logit_scale = 1.0

        # freeze selective parts
        if freeze_visual and hasattr(self.model, "visual"):
            for p in self.model.visual.parameters():
                p.requires_grad = False
        if freeze_text and hasattr(self.model, "transformer"):
            for p in self.model.transformer.parameters():
                p.requires_grad = False

        # token + embedding caches
        self._token_cache = {}
        self._cached_base_class_embeds = None
        self._cached_base_class_embeds_for_templates = None

        if self.verbose:
            print("[CLIPWrapper] CLIP model loaded ✅")
            print(f"  visual_dim={self.visual_embedding_dim}, text_dim={self.text_embedding_dim}")
            if len(self.classnames) > 0:
                print(f"  Using {len(self.classnames)} classnames")

    # ---------- Utilities ----------
    def set_classnames(self, classnames: Iterable[str], clear_cache: bool = True):
        self.classnames = list(map(str, classnames))
        if clear_cache:
            self.clear_token_cache()

    def set_templates(self, templates: Iterable[str], clear_cache: bool = True):
        self.templates = list(templates)
        if clear_cache:
            self.clear_token_cache()

    def clear_token_cache(self):
        self._token_cache.clear()
        self._cached_base_class_embeds = None
        self._cached_base_class_embeds_for_templates = None

    # ---------- Core Tokenization ----------
    def _tokenize_and_cache(self, texts: List[str]):
        key = tuple(texts)
        if key in self._token_cache:
            return self._token_cache[key]
        tok = clip.tokenize(list(texts)).to(self.device)
        self._token_cache[key] = tok
        return tok

    def _batched_forward(self, fn, inputs: torch.Tensor, batch_size: int = 32, desc: Optional[str] = None):
        N = inputs.shape[0]
        outs = []
        for i in range(0, N, batch_size):
            sub = inputs[i:i + batch_size].to(self.device)
            with torch.no_grad():
                out = fn(sub)
            outs.append(out.detach())
        return torch.cat(outs, dim=0).to(self.device).float()

    # ---------- Image Encoding ----------
    def encode_image(self, images: Union[torch.Tensor, Iterable[Image.Image]], batch_size: int = 32):
        if not isinstance(images, torch.Tensor):
            images = torch.stack([self.preprocess(im) for im in images], dim=0)
        images = images.float()
        if images.shape[0] == 0:
            return torch.empty((0, self.visual_embedding_dim), device=self.device, dtype=torch.float)
        return self._batched_forward(self.model.encode_image, images.cpu(), batch_size=batch_size)

    def encode_image_with_prompt(
        self,
        images: Union[torch.Tensor, Iterable[Image.Image]],
        visual_prompt: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        prompt_mode: str = "add_mean",
    ):
        img_emb = self.encode_image(images, batch_size=batch_size)
        if visual_prompt is None:
            return img_emb
        prompt = visual_prompt.to(self.device).float()
        prompt_avg = prompt.mean(dim=0, keepdim=True)
        if prompt_avg.shape[-1] != img_emb.shape[-1]:
            raise ValueError(f"visual_prompt dim {prompt_avg.shape[-1]} != image embedding dim {img_emb.shape[-1]}")
        return img_emb + prompt_avg

    # ---------- Text Encoding ----------
    def encode_text(self, texts: List[str], batch_size: int = 32):
        if len(texts) == 0:
            return torch.empty((0, self.text_embedding_dim), device=self.device)
        tokenized = self._tokenize_and_cache(texts)
        return self._batched_forward(self.model.encode_text, tokenized.cpu(), batch_size=batch_size)

    def _compute_and_cache_base_class_embeds(self, batch_size: int = 32):
        if len(self.classnames) == 0:
            raise ValueError("CLIPWrapper: classnames is empty.")
        templates_key = tuple(self.templates)
        if (
            self._cached_base_class_embeds is not None
            and self._cached_base_class_embeds_for_templates == templates_key
        ):
            return
        all_template_embeds = []
        for template in self.templates:
            texts = [template.format(c) for c in self.classnames]
            emb = self.encode_text(texts, batch_size=batch_size)
            all_template_embeds.append(emb)
        if len(all_template_embeds) == 1:
            base_embeds = all_template_embeds[0]
        else:
            base_embeds = torch.stack(all_template_embeds, dim=0).mean(dim=0)
        self._cached_base_class_embeds = base_embeds.to(self.device)
        self._cached_base_class_embeds_for_templates = templates_key

    def encode_text_with_prompt(
        self,
        classnames: Optional[List[str]] = None,
        text_prompt: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ):
        if classnames is None:
            classnames = self.classnames
        else:
            classnames = list(map(str, classnames))
        if (
            self._cached_base_class_embeds is None
            or self._cached_base_class_embeds_for_templates != tuple(self.templates)
        ):
            self._compute_and_cache_base_class_embeds(batch_size=batch_size)
        base = self._cached_base_class_embeds
        if text_prompt is None:
            return base
        prompt = text_prompt.to(self.device).float()
        prompt_avg = prompt.mean(dim=0, keepdim=True)
        if prompt_avg.shape[-1] != base.shape[-1]:
            raise ValueError(f"text_prompt dim {prompt_avg.shape[-1]} != text embed dim {base.shape[-1]}")
        return base + prompt_avg

    # ---------- helpers ----------
    @property
    def device_str(self):
        return self.device

    @property
    def dtype(self):
        return torch.get_default_dtype()


# ---------- Dataset Classname Utility ----------
def get_class_names_from_dataset(dataset_obj):
    if hasattr(dataset_obj, "class_names") and dataset_obj.class_names is not None:
        return list(dataset_obj.class_names)
    if hasattr(dataset_obj, "classes") and dataset_obj.classes is not None:
        return list(dataset_obj.classes)
    for attr in ("num_classes", "n_classes", "N_CLASSES", "n_labels"):
        if hasattr(dataset_obj, attr):
            n = int(getattr(dataset_obj, attr))
            return [str(i) for i in range(n)]
    if hasattr(dataset_obj, "dataset") and dataset_obj.dataset is not dataset_obj:
        return get_class_names_from_dataset(dataset_obj.dataset)
    if hasattr(dataset_obj, "datasets"):
        for d in dataset_obj.datasets:
            try:
                names = get_class_names_from_dataset(d)
                if names:
                    return names
            except Exception:
                continue
    try:
        if hasattr(dataset_obj, "targets") and dataset_obj.targets is not None:
            labels = dataset_obj.targets
            n = int(max(labels)) + 1
            return [str(i) for i in range(n)]
    except Exception:
        pass
    raise ValueError("Could not infer class names from dataset.")
