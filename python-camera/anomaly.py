# -*- coding: utf-8 -*-
"""
PatchCore 精简版异常检测（CPU 可跑，无需 GPU）。

思想：用 ImageNet 预训练的 ResNet 提取图像中层特征，把所有“合格样本”各空间位置的
特征向量存进“记忆库”。检测时，待测图每个位置的特征到记忆库找最近邻距离——
距离大 = 和所有合格样本都不像 = 异常（改动/遮挡/缺件）。取距离图最大值作为图像异常分。

用法：
  1) 安装依赖（CPU 版 PyTorch）：
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       pip install numpy opencv-python
  2) 固定成像后，拍 30~50 张【合格件】图片放到一个目录，例如 ok_images/
  3) 训练（建记忆库）：
       python anomaly.py train --ok-dir ok_images --out mem.npz
  4) 测试单张（打印异常分 + 存热力图）：
       python anomaly.py test --mem mem.npz --image 待测.jpg --out result.jpg
  5) 批量测试一个目录，看分数分布：
       python anomaly.py test --mem mem.npz --dir test_images

判定：异常分 > 阈值(threshold) => NG。阈值在 train 时由合格样本自身分数自动estimate，
也可在 test 时用 --threshold 手动覆盖。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchvision
except ImportError:
    print("缺少依赖。请先安装：\n"
          "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
          "  pip install numpy opencv-python")
    sys.exit(1)

INPUT = 224          # 网络输入边长
FEAT_SIZE = 28       # 特征图统一到 28x28
CORESET = 12000      # 记忆库最大向量数（随机下采样，控制速度/内存）
torch.set_num_threads(max(1, os.cpu_count() or 1))

_model = None


def get_model():
    """加载预训练 ResNet18，取 layer2+layer3 中层特征（首次运行会联网下载权重）。"""
    global _model
    if _model is None:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.eval()
        from torchvision.models.feature_extraction import create_feature_extractor
        _model = create_feature_extractor(net, return_nodes={"layer2": "l2", "layer3": "l3"})
    return _model


def embed(img_bgr):
    """图像 -> 每个空间位置的特征向量。返回 (N, D) 与空间尺寸 FEAT_SIZE。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)
    with torch.no_grad():
        feats = get_model()(x)
    # 两层特征插值到同尺寸后拼接（多尺度）
    parts = []
    for key in ("l2", "l3"):
        f = F.interpolate(feats[key], size=(FEAT_SIZE, FEAT_SIZE), mode="bilinear", align_corners=False)
        parts.append(f)
    emb = torch.cat(parts, dim=1)[0]                    # (D, H, W)
    d = emb.shape[0]
    emb = emb.permute(1, 2, 0).reshape(-1, d).numpy()   # (H*W, D)
    return emb.astype(np.float32)


def list_images(path):
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(path, ext))
            files += glob.glob(os.path.join(path, ext.upper()))
        return sorted(files)
    return [path]


def build_memory(ok_dir):
    files = list_images(ok_dir)
    if not files:
        raise RuntimeError("目录里没有图片: " + ok_dir)
    print("合格样本 %d 张，提取特征中…" % len(files))
    all_emb = []
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None:
            continue
        all_emb.append(embed(img))
        print("  [%d/%d] %s" % (i + 1, len(files), os.path.basename(fp)))
    mem = np.concatenate(all_emb, axis=0)
    # 随机下采样成 coreset，控制记忆库规模
    if mem.shape[0] > CORESET:
        idx = np.random.RandomState(0).choice(mem.shape[0], CORESET, replace=False)
        mem = mem[idx]
    print("记忆库向量数: %d，维度: %d" % (mem.shape[0], mem.shape[1]))
    return mem, files


def score_image(img_bgr, mem_t):
    """返回 (图像异常分, 异常热力图[INPUT×INPUT])。"""
    emb = torch.from_numpy(embed(img_bgr))              # (P, D)
    dists = torch.cdist(emb, mem_t)                     # (P, M)
    mind = dists.min(dim=1).values                      # 每位置到记忆库最近距离
    amap = mind.reshape(FEAT_SIZE, FEAT_SIZE).numpy()
    amap = cv2.resize(amap, (INPUT, INPUT), interpolation=cv2.INTER_CUBIC)
    amap = cv2.GaussianBlur(amap, (0, 0), 4)
    # 图像级分数：取热力图高分位（比单点最大更稳）
    img_score = float(np.percentile(amap, 99.5))
    return img_score, amap


def make_heatmap(img_bgr, amap, vmax):
    small = cv2.resize(img_bgr, (INPUT, INPUT))
    norm = np.clip(amap / (vmax + 1e-6), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(small, 0.6, heat, 0.4, 0)


def cmd_train(args):
    mem, files = build_memory(args.ok_dir)
    mem_t = torch.from_numpy(mem)
    # 用合格样本自身估计正常分数分布 -> 阈值
    print("估计正常分数分布…")
    scores = []
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        s, _ = score_image(img, mem_t)
        scores.append(s)
    scores = np.array(scores)
    mean, std, mx = scores.mean(), scores.std(), scores.max()
    threshold = float(mx + 2 * std)   # 保守阈值：最高正常分再加 2 个标准差
    print("正常分数: 均值=%.3f 标准差=%.3f 最大=%.3f => 阈值=%.3f" % (mean, std, mx, threshold))
    np.savez_compressed(args.out, memory=mem, threshold=threshold,
                        train_max=mx, train_mean=mean, train_std=std)
    print("已保存记忆库: %s" % args.out)


def cmd_test(args):
    data = np.load(args.mem)
    mem_t = torch.from_numpy(data["memory"])
    threshold = float(args.threshold) if args.threshold is not None else float(data["threshold"])
    print("阈值 = %.3f（超过判 NG）" % threshold)

    files = list_images(args.dir if args.dir else args.image)
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            print("跳过无法读取: %s" % fp)
            continue
        s, amap = score_image(img, mem_t)
        verdict = "NG" if s > threshold else "OK"
        print("%-40s 异常分=%.3f  => %s" % (os.path.basename(fp), s, verdict))
        if args.out and not args.dir:
            heat = make_heatmap(img, amap, max(threshold, s))
            cv2.imwrite(args.out, heat)
            print("热力图已存: %s" % args.out)


def main():
    ap = argparse.ArgumentParser(description="PatchCore 精简版异常检测（CPU）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="用合格样本建记忆库")
    t.add_argument("--ok-dir", required=True, help="合格样本图片目录")
    t.add_argument("--out", default="mem.npz", help="输出记忆库文件")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("test", help="检测单张或一个目录")
    e.add_argument("--mem", default="mem.npz", help="记忆库文件")
    e.add_argument("--image", help="单张待测图")
    e.add_argument("--dir", help="批量测试目录")
    e.add_argument("--out", help="单张时输出热力图路径")
    e.add_argument("--threshold", type=float, default=None, help="手动覆盖阈值")
    e.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
