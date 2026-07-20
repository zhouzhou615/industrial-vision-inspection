package com.vision.inspect.compare;

import com.vision.inspect.model.RoiRegion;
import org.opencv.core.Mat;

import java.util.Optional;

/**
 * 图像比对算法抽象。可扩展 SSIM / 模板匹配 / 深度学习等不同实现。
 */
public interface ImageComparator {
    CompareScore compare(Mat template, Mat captured, Optional<RoiRegion> roi);
}
