package com.vision.inspect.detect;

import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * 在标准图上自动识别螺丝位（螺丝头/圆孔近似圆形）。
 * 采用轮廓+最小外接圆的方式，对手绘细线圈、椭圆孔比霍夫圆更稳，
 * 并按半径范围过滤掉过大(如 Logo 方框)/过小(噪点)的轮廓。
 */
public final class ScrewAutoDetector {

    private ScrewAutoDetector() {
    }

    /**
     * @param templateColor 标准图（BGR）
     * @param minR 最小螺丝半径（像素）
     * @param maxR 最大螺丝半径（像素）
     */
    public static List<ScrewPoint> detect(Mat templateColor, int minR, int maxR) {
        Mat gray = new Mat();
        Imgproc.cvtColor(templateColor, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        // 自适应阈值：深色线条/孔 -> 白，适应不均匀光照
        Mat bin = new Mat();
        Imgproc.adaptiveThreshold(gray, bin, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 31, 10);
        // 形态学闭运算，连接断裂的圈
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(bin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<double[]> raw = new ArrayList<>(); // [x, y, r]
        for (MatOfPoint c : contours) {
            MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
            Point center = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(c2f, center, radius);
            c2f.release();
            double r = radius[0];
            if (r < minR || r > maxR) {
                continue;
            }
            double area = Imgproc.contourArea(c);
            // 过滤过于细碎/不成形的轮廓（面积至少为外接圆的 8%）
            if (area < 0.08 * Math.PI * r * r) {
                continue;
            }
            raw.add(new double[]{center.x, center.y, r});
        }

        // 去重：中心距离小于 minR 视为同一处
        List<double[]> merged = new ArrayList<>();
        for (double[] p : raw) {
            boolean dup = false;
            for (double[] q : merged) {
                if (Math.hypot(p[0] - q[0], p[1] - q[1]) < minR) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                merged.add(p);
            }
        }

        // 按行(y)再列(x)排序编号
        int rowH = Math.max(1, 2 * maxR);
        merged.sort((a, b) -> {
            int ra = (int) (a[1] / rowH), rb = (int) (b[1] / rowH);
            if (ra != rb) {
                return Integer.compare(ra, rb);
            }
            return Double.compare(a[0], b[0]);
        });

        List<ScrewPoint> points = new ArrayList<>();
        for (int i = 0; i < merged.size(); i++) {
            double[] p = merged.get(i);
            points.add(new ScrewPoint("S" + (i + 1),
                    (int) Math.round(p[0]), (int) Math.round(p[1]),
                    (int) Math.round(Math.max(minR, p[2] + 2))));
        }

        gray.release();
        bin.release();
        kernel.release();
        return points;
    }
}
