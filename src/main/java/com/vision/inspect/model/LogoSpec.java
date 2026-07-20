package com.vision.inspect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Logo 检测区定义（标准图像素坐标）及判定参数。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LogoSpec {
    private int x;
    private int y;
    private int width;
    private int height;
    /** 允许的最大歪斜角度（度），超过判 NG */
    private double maxSkewDeg = 8.0;
    /** 判定为“正确 Logo”所需的最少特征匹配点数（仅在特征足够时参与歪斜估计） */
    private int minGoodMatches = 12;
    /** Logo 模板匹配得分阈值(0~1)，低于此判为 Logo 错误/缺失（主判据，更鲁棒） */
    private double minScore = 0.45;
    /** “最大单块连续变化”占区域比例的阈值，超过判为 Logo 被改动/遮挡。
     *  用最大连通块而非总占比：光照/错位的零散噪点被滤除，故阈值可较小。默认 0.03(3%)。 */
    private double maxDiffRatio = 0.03;
}
