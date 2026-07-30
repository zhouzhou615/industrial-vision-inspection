package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 螺丝漏打检测（逐颗模板匹配，不依赖整图对齐）。
 *
 * <p>分两段，两段用<b>不同大小</b>的模板，这一点是关键：</p>
 * <ol>
 *   <li><b>定位</b>：用「螺丝 + 周围板面网点」的<b>大</b>邻域(约 4 倍螺丝半径)在采图里匹配，
 *       得到每颗螺丝<b>自己</b>的真实位置。大邻域里的网点图案是唯一的，所以定位很准；
 *       而且螺丝就算掉了，周围网点仍能把位置定出来。</li>
 *   <li><b>判在位</b>：在定位到的位置上，只拿<b>螺丝本体</b>的小 patch 比外观，
 *       取灰度与边缘两路得分的<b>较小值</b>。</li>
 * </ol>
 *
 * <p>为什么必须这么分（都是实测结论，别再改回去）：</p>
 * <ul>
 *   <li>小 patch 定位不了：7px 的螺丝小图在板上到处都能匹配，实测某颗螺丝的小图
 *       在<b>错误位置</b>拿到 0.967、比正确位置还高。大邻域则 6/6 全部定位正确。</li>
 *   <li>大 patch 判不了在位：邻域被周围网点主导，螺丝掉了得分照样 0.91。</li>
 *   <li>不能用整体平移把示教点阵套上去：工件轻微旋转/透视会让各颗的位移不一样
 *       （实测同一件上从 +200px 渐变到 +166px，差 34px），一个平移量套不住所有点；
 *       示教点又常常近似共线（实测次轴散布只有主轴的 5%），拟合仿射是病态问题，
 *       会把端点那颗甩出 100px 以上，造成假漏打。故改为每颗各自定位。</li>
 *   <li>灰度与边缘不能取<b>大</b>值：空孔和螺丝都是暗斑，灰度分几乎一样高
 *       （实测空孔 0.85 vs 有螺丝 0.94），只有边缘分能区分（空孔 0.63 vs 有螺丝 0.85~0.94）。
 *       取大值等于让空孔用灰度分蒙混过关。</li>
 *   <li>传进来的灰度图<b>不要先做 equalizeHist</b>：全局均衡按整帧直方图重映射，
 *       标准图与采图露出的暗背景多少不同，两边就被映射成不同灰度，小 patch 的归一化相关随之失真。
 *       实测均衡后「有螺丝最低 0.66 / 空孔最高 0.65」几乎不可分，用原始灰度则是「0.88 / 0.72」。
 *       CLAHE 同样更差（0.78/0.79）。TM_CCOEFF_NORMED 本身已对整体明暗不敏感，不需要预先均衡。</li>
 * </ul>
 *
 * <p>按上述实测，本产品(螺丝直径约 14px)的 {@code screwMinScore} 取 <b>0.80</b> 最稳：
 * 有螺丝 0.87~0.99、空孔 0.32~0.72，两侧都有余量。</p>
 */
public final class ScrewPatternDetector {

    private static final Logger log = LoggerFactory.getLogger(ScrewPatternDetector.class);

    /** 定位邻域半径 = 螺丝半径 × 此倍数（并不低于 {@link #MIN_CONTEXT_R}）。 */
    private static final int CONTEXT_SCALE = 4;
    private static final int MIN_CONTEXT_R = 24;
    /** 定位匹配分下限：低于此说明该处邻域整体不像标准图（遮挡/超出视野/严重失焦）。 */
    private static final double MIN_LOCATE_SCORE = 0.5;

    /** 检测结果：缺陷列表 + 工件整体位移（中位数，用于把示教坐标画到采图真实位置）。 */
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
     */
    public static Result detect(Mat templateGray, Mat alignedGray, InspectionSpec spec) {
        List<Defect> defects = new ArrayList<>();
        List<ScrewPoint> screws = spec.getScrews();
        if (screws == null || screws.isEmpty()) {
            return new Result(defects, 0, 0);
        }
        int rBase = medianRadius(screws);
        double minScore = spec.getScrewMinScore() > 0 ? spec.getScrewMinScore() : 0.6;
        int n = screws.size();

        // 1. 粗定位：在半分辨率上全图找，只为拿一个工件大致位移（半分辨率快 4 倍，精度足够）
        double[] coarse = coarseShift(templateGray, alignedGray, screws);

        // 2. 精定位：在「示教位置 + 粗位移」附近的窗口内全分辨率匹配，得到每颗自己的位置。
        //    限定窗口很重要：不限定时大邻域偶尔会匹配到板上别处（实测跳 322px）。
        int win = Math.max(40, rBase * 5);
        double[] offX = new double[n];
        double[] offY = new double[n];
        double[] locScore = new double[n];
        for (int i = 0; i < n; i++) {
            double[] l = locate(templateGray, alignedGray, screws.get(i), rBase,
                    coarse[0], coarse[1], win);
            offX[i] = l[0];
            offY[i] = l[1];
            locScore[i] = l[2];
        }
        double mdx = median(offX);
        double mdy = median(offY);

        // 位置判据的容差要容得下工件真实的轻微旋转/透视（各颗位移本来就不完全相同），
        // 否则边角那颗会被误判；「螺丝挪位」主要靠外观判据兜住。
        double tolPos = Math.max(30, rBase * 4);
        double maxDev = 0;
        for (int i = 0; i < n; i++) {
            maxDev = Math.max(maxDev, Math.hypot(offX[i] - mdx, offY[i] - mdy));
        }
        log.info("  工件位移 dx={} dy={} (各颗最大偏离中位 {}px, 位置容差 {}px)",
                Math.round(mdx), Math.round(mdy), Math.round(maxDev), Math.round(tolPos));

        // 半数以上螺丝同时"位置偏移"或"定位分过低"，几乎不可能是螺丝本身的问题
        // （真打偏只会是个别颗），而是<b>整图对齐没把工件转正</b>：本检测按纯平移处理，
        // 残留旋转会让各颗位移差出几十像素。实测对齐残留 -7° 时就是这个症状。
        int suspect = 0;
        for (int i = 0; i < n; i++) {
            if (Math.hypot(offX[i] - mdx, offY[i] - mdy) > tolPos || locScore[i] < MIN_LOCATE_SCORE) {
                suspect++;
            }
        }
        if (suspect * 2 > n) {
            log.warn("  ⚠ {}/{} 颗同时位置异常/定位分过低 —— 这通常不是螺丝问题，而是整图对齐失败"
                    + "（工件残留旋转）。请检查 ImageAligner 是否把工件转正、以及工件是否被遮挡或超出视野。",
                    suspect, n);
        }

        // 3. 逐颗判定
        for (int i = 0; i < n; i++) {
            ScrewPoint sp = screws.get(i);
            double ex = sp.getX() + offX[i];
            double ey = sp.getY() + offY[i];
            double dev = Math.hypot(offX[i] - mdx, offY[i] - mdy);
            double[] sc = presenceScores(templateGray, alignedGray, sp, ex, ey, rBase);
            double score = Math.min(sc[0], sc[1]);

            boolean locOk = locScore[i] >= MIN_LOCATE_SCORE;
            boolean posOk = dev <= tolPos;
            boolean present = locOk && posOk && score >= minScore;
            log.info("  螺丝 {} 定位偏离={}px(定位分{}) 灰度={} 边缘={} 判据={}(阈值{}) -> {}",
                    sp.getId(), String.format("%.1f", dev), String.format("%.2f", locScore[i]),
                    String.format("%.2f", sc[0]), String.format("%.2f", sc[1]),
                    String.format("%.2f", score), minScore, present ? "有螺丝" : "漏打");
            if (!present) {
                String why;
                if (!locOk) {
                    why = String.format("该处邻域与标准图不符(定位分%.2f<%.2f),可能被遮挡或超出视野",
                            locScore[i], MIN_LOCATE_SCORE);
                } else if (!posOk) {
                    why = String.format("位置偏移%.0fpx>%.0fpx,疑似打偏", dev, tolPos);
                } else {
                    why = String.format("外观不符,疑似空孔(得分%.2f<%.2f)", score, minScore);
                }
                defects.add(missing(sp, ex - sp.getX(), ey - sp.getY(), rBase, why));
            }
        }
        return new Result(defects, mdx, mdy);
    }

    /**
     * 半分辨率全图粗定位，返回工件位移中位数。
     * 只用来给精定位提供搜索窗中心，所以半分辨率的精度（±2px）完全够用。
     */
    private static double[] coarseShift(Mat templateGray, Mat alignedGray, List<ScrewPoint> screws) {
        Mat tHalf = new Mat();
        Mat aHalf = new Mat();
        Imgproc.resize(templateGray, tHalf, new Size(), 0.5, 0.5, Imgproc.INTER_AREA);
        Imgproc.resize(alignedGray, aHalf, new Size(), 0.5, 0.5, Imgproc.INTER_AREA);
        try {
            int n = screws.size();
            double[] cx = new double[n];
            double[] cy = new double[n];
            for (int i = 0; i < n; i++) {
                ScrewPoint sp = screws.get(i);
                ScrewPoint half = new ScrewPoint(sp.getId(), sp.getX() / 2, sp.getY() / 2,
                        Math.max(2, sp.getR() / 2));
                double[] l = locate(tHalf, aHalf, half, Math.max(2, sp.getR() / 2), 0, 0, -1);
                cx[i] = l[0] * 2;
                cy[i] = l[1] * 2;
            }
            return new double[]{median(cx), median(cy)};
        } finally {
            tHalf.release();
            aHalf.release();
        }
    }

    /**
     * 用「螺丝 + 周围网点」的大邻域定位一颗螺丝。
     *
     * @param win 搜索窗半宽；&lt;0 表示整张采图搜索
     * @return {dx, dy, 匹配分}
     */
    private static double[] locate(Mat templateGray, Mat alignedGray, ScrewPoint sp, int rBase,
                                   double gdx, double gdy, int win) {
        int r = Math.max(4, sp.getR() > 0 ? sp.getR() : rBase);
        int rc = Math.max(MIN_CONTEXT_R, r * CONTEXT_SCALE);
        Rect pr = clamp(sp.getX() - rc, sp.getY() - rc, 2 * rc, 2 * rc,
                templateGray.cols(), templateGray.rows());
        Mat patch = new Mat(templateGray, pr);
        Mat area = null;
        Mat res = new Mat();
        try {
            int offX = 0;
            int offY = 0;
            Mat search;
            if (win > 0) {
                int side = pr.width + 2 * win;
                Rect sr = clamp((int) Math.round(sp.getX() + gdx) - side / 2,
                        (int) Math.round(sp.getY() + gdy) - side / 2,
                        side, side, alignedGray.cols(), alignedGray.rows());
                area = new Mat(alignedGray, sr);
                search = area;
                offX = sr.x;
                offY = sr.y;
            } else {
                search = alignedGray;
            }
            if (search.cols() < pr.width || search.rows() < pr.height) {
                return new double[]{gdx, gdy, 0};
            }
            Imgproc.matchTemplate(search, patch, res, Imgproc.TM_CCOEFF_NORMED);
            Core.MinMaxLocResult mm = Core.minMaxLoc(res);
            double bx = offX + mm.maxLoc.x + pr.width / 2.0;
            double by = offY + mm.maxLoc.y + pr.height / 2.0;
            return new double[]{bx - sp.getX(), by - sp.getY(), mm.maxVal};
        } catch (Exception e) {
            log.warn("  螺丝 {} 定位失败: {}", sp.getId(), e.getMessage());
            return new double[]{gdx, gdy, 0};
        } finally {
            patch.release();
            if (area != null) {
                area.release();
            }
            res.release();
        }
    }

    /**
     * 在已定位的位置上判「有螺丝 / 空孔」：只比螺丝本体，灰度与边缘各一路。
     * 搜索余量只留几像素——精定位已经把位置定准了，余量大反而会把旁边的特征找来充数。
     *
     * @return {灰度分, 边缘分}
     */
    private static double[] presenceScores(Mat templateGray, Mat alignedGray, ScrewPoint sp,
                                           double ex, double ey, int rBase) {
        int r = Math.max(4, sp.getR() > 0 ? sp.getR() : rBase);
        final int margin = 3;
        Rect pr = clamp(sp.getX() - r, sp.getY() - r, 2 * r, 2 * r,
                templateGray.cols(), templateGray.rows());
        int side = 2 * r + 2 * margin;
        int sx = (int) Math.round(ex) - r - margin;
        int sy = (int) Math.round(ey) - r - margin;
        sx = Math.max(0, Math.min(sx, Math.max(0, alignedGray.cols() - side)));
        sy = Math.max(0, Math.min(sy, Math.max(0, alignedGray.rows() - side)));
        Rect sr = clamp(sx, sy, side, side, alignedGray.cols(), alignedGray.rows());
        if (pr.width < 4 || pr.height < 4 || sr.width < pr.width || sr.height < pr.height) {
            log.warn("  螺丝 {} 外观比对窗口无效(patch {}x{}, search {}x{})，按不合格处理",
                    sp.getId(), pr.width, pr.height, sr.width, sr.height);
            return new double[]{0, 0};
        }
        Mat patch = new Mat(templateGray, pr);
        Mat search = new Mat(alignedGray, sr);
        Mat pe = EdgeUtil.gradient(patch);
        Mat se = EdgeUtil.gradient(search);
        try {
            return new double[]{EdgeUtil.matchScore(search, patch), EdgeUtil.matchScore(se, pe)};
        } finally {
            patch.release();
            search.release();
            pe.release();
            se.release();
        }
    }

    private static double median(double[] v) {
        double[] c = v.clone();
        java.util.Arrays.sort(c);
        return c[c.length / 2];
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

    private static int medianRadius(List<ScrewPoint> screws) {
        List<Integer> rs = new ArrayList<>();
        for (ScrewPoint s : screws) {
            rs.add(Math.max(3, s.getR()));
        }
        rs.sort(Integer::compareTo);
        return rs.get(rs.size() / 2);
    }
}
