import random
import torch

def paired_patch_augmentation(images, masks, patch_size=64, layout='2x2'):
    """
    参数：
        images: list of image tensors, 每个 tensor 是 (C, H, W)
        masks:  list of mask tensors, 每个 tensor 是 (1, H, W)
        patch_size: 每个 patch 的尺寸
        layout: 拼接布局，目前仅支持 '2x2'
    返回：
        new_image: 拼接后的图像 tensor
        new_mask: 拼接后的掩码 tensor
    """
    assert len(images) == len(masks), "图像与mask数量必须一致"
    assert layout == '2x2', "目前仅支持2x2拼接"

    patch_imgs = []
    patch_masks = []

    for img, mask in zip(images, masks):
        _, H, W = img.shape
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)

        patch_img = img[:, top:top + patch_size, left:left + patch_size]
        patch_mask = mask[:, top:top + patch_size, left:left + patch_size]

        patch_imgs.append(patch_img)
        patch_masks.append(patch_mask)

    # 拼接图像：2x2 格式
    top_img = torch.cat(patch_imgs[:2], dim=2)
    bottom_img = torch.cat(patch_imgs[2:], dim=2)
    new_image = torch.cat([top_img, bottom_img], dim=1)

    top_mask = torch.cat(patch_masks[:2], dim=2)
    bottom_mask = torch.cat(patch_masks[2:], dim=2)
    new_mask = torch.cat([top_mask, bottom_mask], dim=1)

    return new_image, new_mask
