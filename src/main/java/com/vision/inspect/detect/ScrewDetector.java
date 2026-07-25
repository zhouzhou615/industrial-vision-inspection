package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 螺丝漏打检测（两段式，抓住"有无螺丝"的本质差异）：
 *
 * <ol>
 *   <li><b>先定位</b>：用整个螺丝框（含周围板面，特征多）在小搜索窗内做梯度匹配，
 *       精确找到螺丝<b>应该在的位置</b>，补偿残余对齐误差。</li>
 *   <li><b>再判有无</b>：在定位好的点上，<b>只比对中心那一小块螺丝头</b>（灰度/边缘取优）。
 *       装了螺丝→中心匹配得分高；漏打→中心是空洞/背景，得分骤降。</li>
 * </ol>
 *
 * <p>关键：判定只看中心小块，不看周围板面 —— 这样"装没装螺丝"才有明显区分，
 * 不会被大片相同的周围背景稀释掉。</p>
 */
public final class ScrewDetector {

    private static final Logger log = LoggerFactory.getLogger(ScrewDetector.class);

    private ScrewDetector() {
    }

    public static List<Defect> detect(Mat templateGray, Mat alignedGray, InspectionSpec spec) {
        List<Defect> defects = new ArrayList<>();
        int cols = templateGray.cols();
        int rows = templateGray.rows();

        Mat tGrad = EdgeUtil.gradient(templateGray);
        Mat cGrad = EdgeUtil.gradient(alignedGray);
        try {
            for (ScrewPoint sp : spec.getScrews()) {
                int r = Math.max(4, sp.getR());

                // ==== 第1段：用整框定位螺丝的真实位置（补偿残余对齐误差）====
                Rect patchRect = clamp(sp.getX() - r, sp.getY() - r, 2 * r, 2 * r, cols, rows);
                int margin = Math.max(6, (int) (r * 0.6));
                Rect searchRect = clamp(sp.getX() - r - margin, sp.getY() - r - margin,
                        2 * r + 2 * margin, 2 * r + 2 * margin, cols, rows);

                int locCx = sp.getX();  // 定位到的螺丝中心（采图坐标），默认=标称位置
                int locCy = sp.getY();
                if (patchRect.width > 3 && patchRect.height > 3
                        && searchRect.width > patchRect.width && searchRect.height > patchRect.height) {
                    Mat patch = new Mat(tGrad, patchRect);
                    Mat search = new Mat(cGrad, searchRect);
                    Mat result = new Mat();
                    Imgproc.matchTemplate(search, patch, result, Imgproc.TM_CCOEFF_NORMED);
                    Point loc = Core.minMaxLoc(result).maxLoc;
                    // 匹配到的框左上角 → 中心
                    locCx = (int) (searchRect.x + loc.x + patchRect.width / 2.0);
                    locCy = (int) (searchRect.y + loc.y + patchRect.height / 2.0);
                    patch.release();
                    search.release();
                    result.release();
                }

                // ==== 第2段：只看中心小块，判断螺丝在不在 ====
                int cr = Math.max(3, (int) (r * 0.4));    // 中心=螺丝头（越小越聚焦螺丝本身）
                int cpad = 4;                              // 微小容差，吸收亚像素偏差
                Rect tCenter = clamp(sp.getX() - cr, sp.getY() - cr, 2 * cr, 2 * cr, cols, rows);
                Rect cCenter = clamp(locCx - cr - cpad, locCy - cr - cpad,
                        2 * cr + 2 * cpad, 2 * cr + 2 * cpad, cols, rows);

                double centerGray = crop2Match(alignedGray, cCenter, templateGray, tCenter);
                double centerEdge = crop2Match(cGrad, cCenter, tGrad, tCenter);
                double presence = Math.max(centerGray, centerEdge);

                log.info("  螺丝 {} 中心边缘={} 中心灰度={} 取={} (阈值<{}={}判漏打)",
                        sp.getId(), fmt(centerEdge), fmt(centerGray), fmt(presence),
                        spec.getScrewMinScore(), presence < spec.getScrewMinScore() ? "是" : "否");

                if (presence < spec.getScrewMinScore()) {
                    defects.add(Defect.builder()
                            .type("SCREW_MISSING")
                            .shape("CIRCLE")
                            .x(sp.getX())
                            .y(sp.getY())
                            .r(r + 4)
                            .message(String.format("螺丝 %s 漏打/异常 (中心 边缘%.2f 灰度%.2f)",
                                    sp.getId() == null ? "?" : sp.getId(), centerEdge, centerGray))
                            .build());
                }
            }
            return defects;
        } finally {
            tGrad.release();
            cGrad.release();
        }
    }

    /** 裁出 search/patch 子图后匹配，返回最大归一化相关得分。 */
    private static double crop2Match(Mat searchImg, Rect searchRect, Mat patchImg, Rect patchRect) {
        Mat patch = new Mat(patchImg, patchRect);
        Mat search = new Mat(searchImg, searchRect);
        try {
            return EdgeUtil.matchScore(search, patch);
        } finally {
            patch.release();
            search.release();
        }
    }

    private static String fmt(double v) {
        return String.format("%.2f", v);
    }

    private static Rect clamp(int x, int y, int w, int h, int cols, int rows) {
        int nx = Math.max(0, Math.min(x, cols - 1));
        int ny = Math.max(0, Math.min(y, rows - 1));
        int nw = Math.max(1, Math.min(w, cols - nx));
        int nh = Math.max(1, Math.min(h, rows - ny));
        return new Rect(nx, ny, nw, nh);
    }
}
