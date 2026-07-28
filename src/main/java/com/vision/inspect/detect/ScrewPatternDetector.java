package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
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
    public static Result detect(Mat templateGray, Mat alignedGray, InspectionSpec spec) {
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
        // 判定容差：实测“有螺丝”残差约 3~17px（纯平移无法完美贴合轻微旋转/透视），
        // 而“真漏打”时最近的只能是较远的网点(58px 起)，故取约 25px 可清晰区分且留足余量。
        double tol = Math.max(25, rBase * 3.5);
        // 配准用宽容差（容忍整体微小偏差，避免丢票导致在歧义解间乱跳）；判定仍用上面的紧容差
        double regTol = Math.max(20, rBase * 2.5);
        // 限制最大位移：工件不会跑出画面一半以上，据此排除“隔一个点阵周期”的歧义解
        double maxShift = Math.max(alignedGray.cols(), alignedGray.rows()) * 0.35;
        double[] shift = bestShift(screws, candidates, regTol, maxShift);
        log.info("  图案配准平移 dx={} dy={} (配准容差{}px, 判定容差{}px, 最大位移{}px)",
                Math.round(shift[0]), Math.round(shift[1]),
                Math.round(regTol), Math.round(tol), Math.round(maxShift));

        // 3. 逐颗核对：位置 + 外观 双判据
        //    位置：附近有没有候选暗斑（无 → 肯定漏打）
        //    外观：与标准图同位置比对（螺丝孔虽也是暗斑，但与“有螺丝”长得不同 → 区分关键）
        double minScore = spec.getScrewMinScore() > 0 ? spec.getScrewMinScore() : 0.6;
        // 3.5 用配准上的螺丝对估计「旋转+缩放+平移」仿射，压掉边缘螺丝的残差
        //     （纯平移无法补偿轻微旋转/透视，离中心越远误差越大）
        Mat affine = estimateAffine(screws, candidates, shift, regTol);

        for (ScrewPoint sp : screws) {
            double[] p = mapPoint(affine, sp.getX(), sp.getY(), shift);
            double ex = p[0];
            double ey = p[1];
            double best = Double.MAX_VALUE;
            for (Point c : candidates) {
                double d = Math.hypot(c.x - ex, c.y - ey);
                if (d < best) {
                    best = d;
                }
            }
            boolean nearOk = best <= tol;
            double score = appearanceScore(templateGray, alignedGray, sp, ex, ey, rBase, tol);
            boolean present = nearOk && score >= minScore;
            log.info("  螺丝 {} 最近候选点={}px(容差{}) 外观得分={}(阈值{}) -> {}",
                    sp.getId(), Math.round(best), Math.round(tol),
                    String.format("%.2f", score), minScore, present ? "有螺丝" : "漏打");
            if (!present) {
                String why = !nearOk
                        ? String.format("该位置无螺丝特征(最近%.0fpx)", best)
                        : String.format("外观不符,疑似空孔(得分%.2f<%.2f)", score, minScore);
                // 缺陷画在采图的真实位置（经仿射映射后的位置）
                defects.add(missing(sp, ex - sp.getX(), ey - sp.getY(), rBase, why));
            }
        }
        return new Result(defects, shift[0], shift[1]);
    }

    /**
     * 外观比对：取标准图该螺丝处的小图，在采图配准位置附近做归一化匹配。
     * 有螺丝 → 与标准图长得一样，得分高；被拆掉只剩空孔 → 外观不同，得分低。
     * 同时用灰度与边缘两路取最优，兼顾光照变化。
     */
    /**
     * 用「配准上的螺丝对」估计部分仿射（旋转+等比缩放+平移）。
     * 纯平移无法补偿轻微旋转/透视，会让边缘螺丝残差偏大；仿射能把残差压到几像素。
     * @return 2x3 仿射矩阵；点对不足或不合理时返回 null（退回纯平移）。
     */
    private static Mat estimateAffine(List<ScrewPoint> screws, List<Point> candidates,
                                      double[] shift, double regTol) {
        List<Point> from = new ArrayList<>();
        List<Point> to = new ArrayList<>();
        for (ScrewPoint sp : screws) {
            double ex = sp.getX() + shift[0];
            double ey = sp.getY() + shift[1];
            double bd = Double.MAX_VALUE;
            Point bp = null;
            for (Point c : candidates) {
                double d = Math.hypot(c.x - ex, c.y - ey);
                if (d < bd) {
                    bd = d;
                    bp = c;
                }
            }
            if (bp != null && bd <= regTol) {
                from.add(new Point(sp.getX(), sp.getY()));
                to.add(bp);
            }
        }
        if (from.size() < 3) {
            return null;
        }
        try {
            MatOfPoint2f f = new MatOfPoint2f(from.toArray(new Point[0]));
            MatOfPoint2f t = new MatOfPoint2f(to.toArray(new Point[0]));
            Mat m = org.opencv.calib3d.Calib3d.estimateAffinePartial2D(f, t);
            f.release();
            t.release();
            if (m == null || m.empty() || m.rows() != 2) {
                return null;
            }
            // 合理性：缩放接近 1、旋转很小，否则多半是误配
            double sc = Math.hypot(m.get(0, 0)[0], m.get(1, 0)[0]);
            if (sc < 0.9 || sc > 1.1) {
                m.release();
                return null;
            }
            return m;
        } catch (Exception e) {
            return null;
        }
    }

    /** 用仿射矩阵映射点；矩阵为空则退回纯平移。 */
    private static double[] mapPoint(Mat m, double x, double y, double[] shift) {
        if (m == null) {
            return new double[]{x + shift[0], y + shift[1]};
        }
        double nx = m.get(0, 0)[0] * x + m.get(0, 1)[0] * y + m.get(0, 2)[0];
        double ny = m.get(1, 0)[0] * x + m.get(1, 1)[0] * y + m.get(1, 2)[0];
        return new double[]{nx, ny};
    }

    private static double appearanceScore(Mat templateGray, Mat alignedGray,
                                          ScrewPoint sp, double ex, double ey, int rBase, double tol) {
        int r = Math.max(4, sp.getR() > 0 ? sp.getR() : rBase);
        int cols = templateGray.cols();
        int rows = templateGray.rows();
        // 标准图 patch
        Rect pr = clamp(sp.getX() - r, sp.getY() - r, 2 * r, 2 * r, cols, rows);
        // 采图搜索窗：余量必须 ≥ 全局配准残差(可达 20px)，否则螺丝落在窗外会误判为“外观不符”。
        // 窗内做局部重定位，得分才真正反映“有无螺丝”而非“对齐误差”。
        int m = (int) Math.max(tol, r * 2);
        Rect sr = clamp((int) Math.round(ex) - r - m,
                (int) Math.round(ey) - r - m,
                2 * r + 2 * m, 2 * r + 2 * m, alignedGray.cols(), alignedGray.rows());
        if (pr.width < 4 || pr.height < 4 || sr.width <= pr.width || sr.height <= pr.height) {
            return 1.0; // 越界无法比对时不误判
        }
        Mat patch = new Mat(templateGray, pr);
        Mat search = new Mat(alignedGray, sr);
        Mat pe = EdgeUtil.gradient(patch);
        Mat se = EdgeUtil.gradient(search);
        try {
            double gray = EdgeUtil.matchScore(search, patch);
            double edge = EdgeUtil.matchScore(se, pe);
            return Math.max(gray, edge);
        } finally {
            patch.release();
            search.release();
            pe.release();
            se.release();
        }
    }

    private static Rect clamp(int x, int y, int w, int h, int cols, int rows) {
        int nx = Math.max(0, Math.min(x, cols - 1));
        int ny = Math.max(0, Math.min(y, rows - 1));
        int nw = Math.max(1, Math.min(w, cols - nx));
        int nh = Math.max(1, Math.min(h, rows - ny));
        return new Rect(nx, ny, nw, nh);
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
    private static double[] bestShift(List<ScrewPoint> screws, List<Point> candidates, double tol,
                                      double maxShift) {
        double bestDx = 0;
        double bestDy = 0;
        int bestVotes = -1;
        double bestErr = Double.MAX_VALUE;
        double bestMag = Double.MAX_VALUE;

        for (ScrewPoint sp : screws) {
            for (Point c : candidates) {
                double dx = c.x - sp.getX();
                double dy = c.y - sp.getY();
                double mag = Math.hypot(dx, dy);
                // 工件不会跑太远：超出允许位移的假设直接丢弃，消除“隔一个点阵周期”的歧义解
                if (mag > maxShift) {
                    continue;
                }
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
                // 票数优先；票数相同时用「误差 + 位移惩罚」比较：
                // 规则点阵会产生“平移一个周期也能对上”的歧义解，其票数与真实解相同，
                // 但位移大得多，故对位移加惩罚，优先选更接近原位的解。
                double cost = err + mag * 0.1;
                double bestCost = bestErr + bestMag * 0.1;
                if (votes > bestVotes || (votes == bestVotes && cost < bestCost)) {
                    bestVotes = votes;
                    bestErr = err;
                    bestMag = mag;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
        // 用所有内点的平均偏移精修，减小单点误差带来的整体偏移
        double sx = 0;
        double sy = 0;
        int n = 0;
        for (ScrewPoint s2 : screws) {
            double ex = s2.getX() + bestDx;
            double ey = s2.getY() + bestDy;
            double bd = Double.MAX_VALUE;
            Point bp = null;
            for (Point c2 : candidates) {
                double d = Math.hypot(c2.x - ex, c2.y - ey);
                if (d < bd) {
                    bd = d;
                    bp = c2;
                }
            }
            if (bp != null && bd <= tol) {
                sx += bp.x - s2.getX();
                sy += bp.y - s2.getY();
                n++;
            }
        }
        if (n >= 2) {
            bestDx = sx / n;
            bestDy = sy / n;
        }
        log.info("  配准命中 {}/{} 颗", bestVotes, screws.size());
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
