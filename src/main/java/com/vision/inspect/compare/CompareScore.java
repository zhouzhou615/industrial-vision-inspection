package com.vision.inspect.compare;

import lombok.Builder;
import lombok.Data;
import org.opencv.core.Mat;

/**
 * 比对得分及差异图。similarity 为综合相似度（0~1）。
 */
@Data
@Builder
public class CompareScore {
    private double similarity;
    private String algorithm;
    private Mat diffImage;
    private String detail;
}
