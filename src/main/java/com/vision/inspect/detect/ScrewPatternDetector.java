package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 螺丝漏打检测（几何图案匹配，不依赖整图对齐）。
 *
 * <p>思路与海康 VisionMaster 的“几何/图案匹配”一致：<b>用螺丝自身当定位特征</b>。</p>
 * <ol>
 *   <li>在采图中找出所有“螺丝状暗斑”作为候选点；</li>
 *   <li>让示教的螺丝点阵与候选点做平移配准（投票法，取匹配最多的平移量）；</li>
 *   <li>配准后逐颗核对：该位置附近有候选点=有螺丝，没有=<b>漏打</b>。</li>
 * </ol>
 * 优点：工件平移、部分出画面都不影响，且能精确指出是哪一颗漏打。
 */
public final class ScrewPatternDetector {

    private static final Logger log = LoggerFactory.getLogger(ScrewPatternDetector.class);

    /** 检测结果：缺陷列表 + 图案配准平移量（用于把示教坐标映射到采图真实位置）。 */
    public static class Result {
        public final List<Defect> defects;
        public final double dx;
        public final double dy;

        Result(List<Defect> defects, double dx, double dy) {
            this.defects = defects;
            this.dx = dx;
            this.dy = dy;
        }
    }

    private ScrewPatternDetector() {
    }

    /**
     * @param alignedGray 采图灰度（未对齐也可）
     * @param spec        检测配置（含示教的螺丝点阵）
     * @return 漏打螺丝的缺陷列表
     */
    public static Result detect(Mat alignedGray, InspectionSpec spec) {
        List<Defect> defects = new ArrayList<>();
        List<ScrewPoint> screws = spec.getScrews();
        if (screws == null || screws.isEmpty()) {
            return new Result(defects, 0, 0);
        }
        // 以示教半径的中位数作为螺丝尺寸基准
        int rBase = medianRadius(screws);

        // 1. 找采图中的螺丝候选点
        List<Point> candidates = findScrewCandidates(alignedGray, rBase);
        log.info("  螺丝候选点 {} 个 (基准半径 {}px)", candidates.size(), rBase);
        if (candidates.isEmpty()) {
            // 一个都找不到：全部判漏打（多为成像异常）
            for (ScrewPoint sp : screws) {
                defects.add(missing(sp, 0, 0, rBase, "采图未找到任何螺丝特征"));
            }
            return new Result(defects, 0, 0);
        }

        // 2. 投票求最佳平移：让示教点阵套到候选点上
        // 判定容差：有螺丝时候选点几乎重合(个位数px)，无螺丝时最近的只能是旁边网点(远得多)，
        // 故取约 1 倍螺丝半径即可清晰区分；过大会让邻近网点冒充螺丝导致漏检。
        double tol = Math.max(6, rBase * 1.0);
        double[] shift = bestShift(screws, candidates, tol);
        log.info("  图案配准平移 dx={} dy={} (容差 {}px)",
                Math.round(shift[0]), Math.round(shift[1]), Math.round(tol));

        // 3. 逐颗核对
        for (ScrewPoint sp : screws) {
            double ex = sp.getX() + shift[0];
            double ey = sp.getY() + shift[1];
            double best = Double.MAX_VALUE;
            for (Point c : candidates) {
                double d = Math.hypot(c.x - ex, c.y - ey);
                if (d < best) {
                    best = d;
                }
            }
            boolean present = best <= tol;
            log.info("  螺丝 {} 最近候选点距离={}px (容差{}px) -> {}",
                    sp.getId(), Math.round(best), Math.round(tol), present ? "有螺丝" : "漏打");
            if (!present) {
                // 缺陷画在采图的真实位置（示教坐标 + 配准平移）
                defects.add(missing(sp, shift[0], shift[1], rBase,
                        String.format("该位置无螺丝特征(最近%.0fpx)", best)));
            }
        }
        return new Result(defects, shift[0], shift[1]);
    }

    private static Defect missing(ScrewPoint sp, double dx, double dy, int rBase, String why) {
        return Defect.builder()
                .type("SCREW_MISSING")
                .shape("CIRCLE")
                .x((int) Math.round(sp.getX() + dx))
                .y((int) Math.round(sp.getY() + dy))
                .r(Math.max(sp.getR(), rBase) + 4)
                .message(String.format("螺丝 %s 漏打 (%s)", sp.getId() == null ? "?" : sp.getId(), why))
                .build();
    }

    /**
     * 找出图中所有“螺丝状暗斑”：自适应阈值取暗区 → 轮廓 → 按半径/圆度筛选。
     */
    private static List<Point> findScrewCandidates(Mat gray, int rBase) {
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
        Mat bin = new Mat();
        // 暗斑 -> 白：块大小随螺丝尺寸自适应
        int block = Math.max(11, (rBase * 6) | 1);   // 保证奇数
        Imgproc.adaptiveThreshold(blur, bin, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, block, 8);
        Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, k);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(bin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double rMin = Math.max(2.0, rBase * 0.35);
        double rMax = rBase * 2.5;
        List<Point> pts = new ArrayList<>();
        for (MatOfPoint c : contours) {
            MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
            Point center = new Point();
            float[] rad = new float[1];
            Imgproc.minEnclosingCircle(c2f, center, rad);
            c2f.release();
            double r = rad[0];
            if (r < rMin || r > rMax) {
                continue;
            }
            double area = Imgproc.contourArea(c);
            // 圆度：面积应接近外接圆，滤掉长条/网点连片
            if (area < 0.35 * Math.PI * r * r) {
                continue;
            }
            pts.add(center);
        }
        blur.release();
        bin.release();
        k.release();
        return pts;
    }

    /**
     * 投票法求最佳平移：枚举“每个示教点 → 每个候选点”的平移假设，
     * 取能让最多示教点找到候选点的那个平移。
     */
    private static double[] bestShift(List<ScrewPoint> screws, List<Point> candidates, double tol) {
        double bestDx = 0;
        double bestDy = 0;
        int bestVotes = -1;
        double bestErr = Double.MAX_VALUE;

        for (ScrewPoint sp : screws) {
            for (Point c : candidates) {
                double dx = c.x - sp.getX();
                double dy = c.y - sp.getY();
                int votes = 0;
                double err = 0;
                for (ScrewPoint s2 : screws) {
                    double ex = s2.getX() + dx;
                    double ey = s2.getY() + dy;
                    double best = Double.MAX_VALUE;
                    for (Point c2 : candidates) {
                        double d = Math.hypot(c2.x - ex, c2.y - ey);
                        if (d < best) {
                            best = d;
                        }
                    }
                    if (best <= tol) {
                        votes++;
                        err += best;
                    }
                }
                // 票数优先，票数相同取总误差小的
                if (votes > bestVotes || (votes == bestVotes && err < bestErr)) {
                    bestVotes = votes;
                    bestErr = err;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
        return new double[]{bestDx, bestDy};
    }

    private static int medianRadius(List<ScrewPoint> screws) {
        List<Integer> rs = new ArrayList<>();
        for (ScrewPoint s : screws) {
            rs.add(Math.max(3, s.getR()));
        }
        rs.sort(Integer::compareTo);
        return rs.get(rs.size() / 2);
    }
}
