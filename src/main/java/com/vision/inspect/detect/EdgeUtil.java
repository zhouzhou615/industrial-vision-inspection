package com.vision.inspect.detect;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * 边缘/梯度工具：把图像转成 Sobel 梯度幅值图，使后续匹配只看形状、
 * 不受整体明暗影响（参考海康 VisionMaster 基于边缘的匹配思路，提升光照鲁棒性）。
 */
final class EdgeUtil {

    private EdgeUtil() {
    }

    /** Sobel 梯度幅值图（8U）。 */
    static Mat gradient(Mat gray) {
        Mat gx = new Mat();
        Mat gy = new Mat();
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3);
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3);
        Mat mag = new Mat();
        Core.magnitude(gx, gy, mag);
        Mat mag8 = new Mat();
        Core.normalize(mag, mag8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        gx.release();
        gy.release();
        mag.release();
        return mag8;
    }

    /** 在 search 图内匹配 patch，返回最大归一化相关得分（TM_CCOEFF_NORMED）。 */
    static double matchScore(Mat search, Mat patch) {
        if (search.cols() < patch.cols() || search.rows() < patch.rows()) {
            return 0.0;
        }
        Mat result = new Mat();
        try {
            Imgproc.matchTemplate(search, patch, result, Imgproc.TM_CCOEFF_NORMED);
            return Core.minMaxLoc(result).maxVal;
        } finally {
            result.release();
        }
    }
}
