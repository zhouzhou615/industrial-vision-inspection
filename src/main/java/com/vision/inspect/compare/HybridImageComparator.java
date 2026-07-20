package com.vision.inspect.compare;

import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.RoiRegion;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;

import java.util.Optional;

/**
 * 混合比对算法：模板匹配定位对齐 + 像素相似度 + 结构相似度（SSIM 近似）。
 *
 * <pre>
 *   综合相似度 = 0.40 * 像素相似度
 *             + 0.35 * 结构相似度(SSIM)
 *             + 0.25 * 模板匹配置信度
 * </pre>
 */
@Component
public class HybridImageComparator implements ImageComparator {

    private final VisionProperties properties;

    public HybridImageComparator(VisionProperties properties) {
        this.properties = properties;
    }

    @Override
    public CompareScore compare(Mat templateColor, Mat capturedColor, Optional<RoiRegion> roi) {
        Mat template = ImagePreProcessor.preprocess(templateColor, roi);
        Mat captured = ImagePreProcessor.preprocess(capturedColor, roi);

        if (template.rows() != captured.rows() || template.cols() != captured.cols()) {
            Mat resized = ImagePreProcessor.resizeToMatch(captured, template);
            captured.release();
            captured = resized;
        }

        Mat aligned = alignByTemplateMatch(template, captured);

        double pixelSim = pixelSimilarity(template, aligned);
        double ssimApprox = structuralSimilarity(template, aligned);
        double templateMatchScore = templateMatchConfidence(template, captured);

        double similarity = 0.40 * pixelSim + 0.35 * ssimApprox + 0.25 * templateMatchScore;

        Mat diff = new Mat();
        Core.absdiff(template, aligned, diff);
        Imgproc.threshold(diff, diff, 25, 255, Imgproc.THRESH_BINARY);

        template.release();
        captured.release();

        return CompareScore.builder()
                .similarity(clamp01(similarity))
                .algorithm(properties.getCompare().getAlgorithm())
                .diffImage(diff)
                .detail(String.format("pixel=%.4f, ssim=%.4f, match=%.4f", pixelSim, ssimApprox, templateMatchScore))
                .build();
    }

    /**
     * 通过模板匹配在待测图中找到与标准图最相似的位置，截取等大区域，补偿小幅位移。
     */
    private Mat alignByTemplateMatch(Mat template, Mat captured) {
        int maxOffset = properties.getCompare().getMaxOffsetPixels();
        if (maxOffset <= 0) {
            return captured.clone();
        }
        int resultCols = captured.cols() - template.cols() + 1;
        int resultRows = captured.rows() - template.rows() + 1;
        if (resultCols <= 0 || resultRows <= 0) {
            return captured.clone();
        }
        Mat result = new Mat();
        Imgproc.matchTemplate(captured, template, result, Imgproc.TM_CCOEFF_NORMED);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        result.release();
        Point matchLoc = mmr.maxLoc;
        int x = (int) Math.max(0, Math.min(matchLoc.x, captured.cols() - template.cols()));
        int y = (int) Math.max(0, Math.min(matchLoc.y, captured.rows() - template.rows()));
        return captured.submat(y, y + template.rows(), x, x + template.cols()).clone();
    }

    /**
     * 像素相似度：基于灰度绝对差均值。
     */
    private double pixelSimilarity(Mat a, Mat b) {
        Mat diff = new Mat();
        Core.absdiff(a, b, diff);
        double meanDiff = Core.mean(diff).val[0] / 255.0;
        diff.release();
        return 1.0 - meanDiff;
    }

    /**
     * 结构相似度（SSIM）近似实现，对光照变化更鲁棒。
     */
    private double structuralSimilarity(Mat a, Mat b) {
        Mat a32 = new Mat();
        Mat b32 = new Mat();
        a.convertTo(a32, CvType.CV_32F);
        b.convertTo(b32, CvType.CV_32F);

        Mat mu1 = new Mat();
        Mat mu2 = new Mat();
        Imgproc.GaussianBlur(a32, mu1, new Size(11, 11), 1.5);
        Imgproc.GaussianBlur(b32, mu2, new Size(11, 11), 1.5);

        Mat mu1Sq = mu1.mul(mu1);
        Mat mu2Sq = mu2.mul(mu2);
        Mat mu1Mu2 = mu1.mul(mu2);

        Mat sigma1Sq = new Mat();
        Mat sigma2Sq = new Mat();
        Mat sigma12 = new Mat();
        Mat tmp1 = new Mat();
        Mat tmp2 = new Mat();

        Core.multiply(a32, a32, tmp1);
        Imgproc.GaussianBlur(tmp1, sigma1Sq, new Size(11, 11), 1.5);
        Core.subtract(sigma1Sq, mu1Sq, sigma1Sq);

        Core.multiply(b32, b32, tmp2);
        Imgproc.GaussianBlur(tmp2, sigma2Sq, new Size(11, 11), 1.5);
        Core.subtract(sigma2Sq, mu2Sq, sigma2Sq);

        Core.multiply(a32, b32, tmp1);
        Imgproc.GaussianBlur(tmp1, sigma12, new Size(11, 11), 1.5);
        Core.subtract(sigma12, mu1Mu2, sigma12);

        double c1 = 0.01 * 0.01;
        double c2 = 0.03 * 0.03;

        Mat ssimMap = new Mat();
        Mat num = new Mat();
        Mat den = new Mat();
        Mat t1 = new Mat();
        Mat t2 = new Mat();

        // t1 = 2*mu1Mu2 + c1 ; t2 = 2*sigma12 + c2 （用 convertTo 做线性变换，避免空 Mat 尺寸不匹配）
        mu1Mu2.convertTo(t1, CvType.CV_32F, 2.0, c1);
        sigma12.convertTo(t2, CvType.CV_32F, 2.0, c2);
        Core.multiply(t1, t2, num);

        Core.add(mu1Sq, mu2Sq, t1);
        Core.add(t1, new Scalar(c1), t1);
        Core.add(sigma1Sq, sigma2Sq, t2);
        Core.add(t2, new Scalar(c2), t2);
        Core.multiply(t1, t2, den);

        Core.divide(num, den, ssimMap);
        double mssim = Core.mean(ssimMap).val[0];

        // 释放临时矩阵
        for (Mat m : new Mat[]{a32, b32, mu1, mu2, mu1Sq, mu2Sq, mu1Mu2,
                sigma1Sq, sigma2Sq, sigma12, tmp1, tmp2, ssimMap, num, den, t1, t2}) {
            m.release();
        }
        return clamp01(mssim);
    }

    /**
     * 模板匹配置信度：在原始待测图中匹配标准图，返回最大归一化相关值。
     */
    private double templateMatchConfidence(Mat template, Mat captured) {
        if (captured.cols() < template.cols() || captured.rows() < template.rows()) {
            Mat resized = ImagePreProcessor.resizeToMatch(captured, template);
            double v = templateMatchConfidence(template, resized);
            resized.release();
            return v;
        }
        Mat result = new Mat();
        Imgproc.matchTemplate(captured, template, result, Imgproc.TM_CCOEFF_NORMED);
        double maxVal = Core.minMaxLoc(result).maxVal;
        result.release();
        return clamp01(maxVal);
    }

    private double clamp01(double v) {
        return Math.max(0.0, Math.min(1.0, v));
    }
}
