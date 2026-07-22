package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.LogoSpec;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

/**
 * Logo 检测：
 *  主判据——模板匹配得分(NCC)：得分低说明 Logo 错误/缺失，稳定可靠；
 *  辅助——特征够多时估计旋转角判歪斜（特征不足则跳过歪斜判定，避免误报）。
 */
public final class LogoInspector {

    public static class LogoResult {
        public boolean passed = true;
        public double skewDeg = 0.0;
        public double score = 1.0;
        public double diffRatio = 0.0;
        public int blobPixels = 0;
        public Defect defect;
    }

    private LogoInspector() {
    }

    public static LogoResult inspect(Mat templateGray, Mat alignedGray, LogoSpec logo) {
        LogoResult res = new LogoResult();
        int cols = templateGray.cols();
        int rows = templateGray.rows();
        Rect rect = clamp(logo.getX(), logo.getY(), logo.getWidth(), logo.getHeight(), cols, rows);
        if (rect.width < 8 || rect.height < 8) {
            return res;
        }
        // 采图侧搜索窗放大 margin，容忍残余对齐误差
        int margin = Math.max(10, Math.min(rect.width, rect.height) / 4);
        Rect searchRect = clamp(rect.x - margin, rect.y - margin,
                rect.width + 2 * margin, rect.height + 2 * margin, cols, rows);

        Mat tLogo = new Mat(templateGray, rect);
        Mat cSearch = new Mat(alignedGray, searchRect);
        // 边缘图版本（抗光照）
        Mat tLogoEdge = EdgeUtil.gradient(tLogo);
        Mat cSearchEdge = EdgeUtil.gradient(cSearch);
        try {
            // 1. 平移定位（边缘匹配，抗光照），得到最佳位置
            org.opencv.core.Point loc = new org.opencv.core.Point(0, 0);
            if (cSearchEdge.cols() >= tLogoEdge.cols() && cSearchEdge.rows() >= tLogoEdge.rows()) {
                Mat r = new Mat();
                Imgproc.matchTemplate(cSearchEdge, tLogoEdge, r, Imgproc.TM_CCOEFF_NORMED);
                loc = Core.minMaxLoc(r).maxLoc;
                r.release();
            }

            // 2. 取最佳位置等大子图 → 局部 ECC 精配准（旋转+错切+缩放也对齐）
            //    之后“匹配得分”和“变化占比”都在精配准图上算：正确但倾斜的 Logo 分数回到高位，
            //    真正换错/遮挡的内容 ECC 对不上，分数低、变化大。
            int w = tLogo.cols();
            int h = tLogo.rows();
            double score = 0.0;
            double diffRatio = 0.0;
            if (cSearch.cols() >= w && cSearch.rows() >= h) {
                int mx = (int) Math.max(0, Math.min(loc.x, cSearch.cols() - w));
                int my = (int) Math.max(0, Math.min(loc.y, cSearch.rows() - h));
                Mat cLogo = cSearch.submat(my, my + h, mx, mx + w).clone();
                Mat reg = registerEcc(tLogo, cLogo);
                Mat regEdge = EdgeUtil.gradient(reg);
                double g = EdgeUtil.matchScore(reg, tLogo);
                double e = EdgeUtil.matchScore(regEdge, tLogoEdge);
                score = Math.max(g, e);
                diffRatio = changedRatio(tLogo, reg, 55);
                regEdge.release();
                reg.release();
                cLogo.release();
            }
            res.score = round2(score);
            res.diffRatio = round2(diffRatio);

            // 3. 旋转角（辅助歪斜判定）
            double skew = estimateSkew(tLogo, new Mat(alignedGray, rect), logo);
            res.skewDeg = round1(skew);

            res.blobPixels = (int) Math.round(diffRatio * (double) rect.width * rect.height);

            // 只用百分比判据：手持/透视下正常残差的“最大变化块”占比稳定为 0%，
            // 真实改动为数个百分点，二者分得很开；绝对像素会被透视噪声干扰，弃用。
            boolean wrong = score < logo.getMinScore();
            boolean altered = diffRatio > logo.getMaxDiffRatio();
            boolean skewed = !Double.isNaN(skew) && Math.abs(skew) > logo.getMaxSkewDeg();
            if (wrong) {
                res.passed = false;
                res.defect = logoDefect(rect, String.format("Logo 错误/缺失 (匹配得分 %.2f < %.2f)",
                        score, logo.getMinScore()), "LOGO_WRONG");
            } else if (altered) {
                res.passed = false;
                res.defect = logoDefect(rect, String.format("Logo 被改动/遮挡 (变化 %.1f%% > %.1f%%)",
                        diffRatio * 100, logo.getMaxDiffRatio() * 100), "LOGO_CHANGED");
            } else if (skewed) {
                res.passed = false;
                res.defect = logoDefect(rect, String.format("Logo 歪斜 %.1f° (限 %.1f°)",
                        skew, logo.getMaxSkewDeg()), "LOGO_SKEW");
            }
            return res;
        } finally {
            tLogo.release();
            cSearch.release();
            tLogoEdge.release();
            cSearchEdge.release();
        }
    }

    /**
     * 局部仿射 ECC 精配准：返回把 cLogo 校正到 tLogo 几何后的图；失败则返回原图副本。
     * ECC 只对齐几何（旋转/错切/缩放/平移），不会掩盖内容改动——改字/遮挡仍留作差异。
     */
    private static Mat registerEcc(Mat tLogo, Mat cLogo) {
        try {
            Mat warp = Mat.eye(2, 3, CvType.CV_32F);
            TermCriteria crit = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 50, 1e-4);
            // 高斯平滑输入有助 ECC 收敛（gaussFiltSize=5）
            Video.findTransformECC(tLogo, cLogo, warp, Video.MOTION_AFFINE, crit, new Mat(), 5);
            Mat out = new Mat();
            Imgproc.warpAffine(cLogo, out, warp, tLogo.size(),
                    Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);
            warp.release();
            return out;
        } catch (Exception e) {
            // ECC 不收敛会抛异常 → 退回未精配准的原图
            return cLogo.clone();
        }
    }

    /**
     * 返回“最大单块连续变化”占区域的比例。
     * 步骤：轻度模糊(容忍错位) → 均值对齐(抵消光照) → 绝对差 → 二值化 → 开运算去零散噪点
     *       → 找连通块，取最大一块的面积占比。
     * 这样光照/错位产生的零散噪点被滤掉，而真实遮挡/改字（成片变化）会被抓出来。
     */
    private static double changedRatio(Mat a, Mat b, int thr) {
        Mat ab = new Mat();
        Mat bb = new Mat();
        Imgproc.GaussianBlur(a, ab, new org.opencv.core.Size(5, 5), 0);
        Imgproc.GaussianBlur(b, bb, new org.opencv.core.Size(5, 5), 0);
        double ma = Core.mean(ab).val[0];
        double mb = Core.mean(bb).val[0];
        if (mb > 1) {
            Core.add(bb, new org.opencv.core.Scalar(ma - mb), bb);
        }
        Mat diff = new Mat();
        Core.absdiff(ab, bb, diff);
        Imgproc.threshold(diff, diff, thr, 255, Imgproc.THRESH_BINARY);
        Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new org.opencv.core.Size(3, 3));
        Imgproc.morphologyEx(diff, diff, Imgproc.MORPH_OPEN, k);
        // 闭运算把邻近变化连成整块，便于取最大连通区
        Imgproc.morphologyEx(diff, diff, Imgproc.MORPH_CLOSE, k);

        List<org.opencv.core.MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(diff, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea = 0;
        for (org.opencv.core.MatOfPoint c : contours) {
            maxArea = Math.max(maxArea, Imgproc.contourArea(c));
            c.release();
        }
        double ratio = maxArea / (double) (diff.rows() * diff.cols());
        ab.release();
        bb.release();
        diff.release();
        k.release();
        return ratio;
    }

    private static Defect logoDefect(Rect rect, String msg, String type) {
        return Defect.builder().type(type).shape("RECT")
                .x(rect.x).y(rect.y).w(rect.width).h(rect.height).message(msg).build();
    }

    /** 用 ORB 匹配估计 Logo 旋转角；特征不足返回 NaN（表示无法判定，跳过歪斜）。 */
    private static double estimateSkew(Mat tLogo, Mat cLogo, LogoSpec logo) {
        try {
            ORB orb = ORB.create(600);
            MatOfKeyPoint kpT = new MatOfKeyPoint();
            MatOfKeyPoint kpC = new MatOfKeyPoint();
            Mat desT = new Mat();
            Mat desC = new Mat();
            orb.detectAndCompute(tLogo, new Mat(), kpT, desT);
            orb.detectAndCompute(cLogo, new Mat(), kpC, desC);
            if (desT.empty() || desC.empty()) {
                cLogo.release();
                return Double.NaN;
            }
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(desT, desC, matches);
            List<DMatch> list = matches.toList();
            KeyPoint[] tk = kpT.toArray();
            KeyPoint[] ck = kpC.toArray();
            List<Point> tp = new ArrayList<>();
            List<Point> cp = new ArrayList<>();
            for (DMatch m : list) {
                if (m.distance < 50) {
                    tp.add(tk[m.queryIdx].pt);
                    cp.add(ck[m.trainIdx].pt);
                }
            }
            cLogo.release();
            if (tp.size() < Math.max(6, logo.getMinGoodMatches())) {
                return Double.NaN; // 特征不足，不判歪斜
            }
            Mat m = Calib3d.estimateAffinePartial2D(
                    new MatOfPoint2f(tp.toArray(new Point[0])),
                    new MatOfPoint2f(cp.toArray(new Point[0])));
            if (m == null || m.empty()) {
                return Double.NaN;
            }
            double a = m.get(0, 0)[0];
            double b = m.get(1, 0)[0];
            m.release();
            return Math.toDegrees(Math.atan2(b, a));
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static Rect clamp(int x, int y, int w, int h, int cols, int rows) {
        int nx = Math.max(0, Math.min(x, cols - 1));
        int ny = Math.max(0, Math.min(y, rows - 1));
        int nw = Math.max(1, Math.min(w, cols - nx));
        int nh = Math.max(1, Math.min(h, rows - ny));
        return new Rect(nx, ny, nw, nh);
    }

    private static double round1(double v) {
        return Double.isNaN(v) ? 0 : Math.round(v * 10.0) / 10.0;
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
