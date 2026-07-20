package com.vision.inspect.compare;

import com.vision.inspect.model.RoiRegion;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Optional;

/**
 * 图像预处理：可选 ROI 裁剪 → 灰度化 → 高斯滤波降噪。
 */
final class ImagePreProcessor {

    private ImagePreProcessor() {
    }

    static Mat preprocess(Mat source, Optional<RoiRegion> roi) {
        Mat work = source.clone();
        if (roi.isPresent()) {
            RoiRegion r = roi.get();
            Rect rect = clampRect(r, work);
            Mat cropped = new Mat(work, rect);
            // 复制出一份独立内存，释放原始 clone，避免持有大图引用
            Mat copy = cropped.clone();
            work.release();
            work = copy;
        }
        Imgproc.cvtColor(work, work, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(work, work, new Size(3, 3), 0);
        return work;
    }

    /**
     * 将 ROI 约束在图像范围内，避免越界异常。
     */
    private static Rect clampRect(RoiRegion r, Mat image) {
        int x = Math.max(0, Math.min(r.getX(), image.cols() - 1));
        int y = Math.max(0, Math.min(r.getY(), image.rows() - 1));
        int w = Math.max(1, Math.min(r.getWidth(), image.cols() - x));
        int h = Math.max(1, Math.min(r.getHeight(), image.rows() - y));
        return new Rect(x, y, w, h);
    }

    static Mat resizeToMatch(Mat source, Mat targetSizeRef) {
        Mat resized = new Mat();
        Imgproc.resize(source, resized, targetSizeRef.size());
        return resized;
    }
}
