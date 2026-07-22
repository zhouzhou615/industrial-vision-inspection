package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 螺丝漏打检测（参考海康 VisionMaster 的做法：基于边缘/梯度的匹配，对光照更鲁棒）。
 *
 * <p>每个螺丝位在采图对应处做两路归一化匹配并取最优：</p>
 * <ol>
 *   <li><b>梯度匹配</b>：先用 Sobel 求梯度幅值再匹配 —— 只看形状/边缘，不受整体明暗影响（主判据）；</li>
 *   <li><b>灰度匹配</b>：原始灰度 NCC，作为补充。</li>
 * </ol>
 * 综合得分低于阈值判为漏打。搜索窗放大以容忍残余对齐误差（相当于逐位局部再定位）。
 */
public final class ScrewDetector {

    private static final Logger log = LoggerFactory.getLogger(ScrewDetector.class);

    private ScrewDetector() {
    }

    public static List<Defect> detect(Mat templateGray, Mat alignedGray, InspectionSpec spec) {
        List<Defect> defects = new ArrayList<>();
        int cols = templateGray.cols();
        int rows = templateGray.rows();

        // 预计算整幅梯度幅值图（边缘特征，光照不变）
        Mat tGrad = EdgeUtil.gradient(templateGray);
        Mat cGrad = EdgeUtil.gradient(alignedGray);
        try {
            for (ScrewPoint sp : spec.getScrews()) {
                int r = Math.max(4, sp.getR());
                Rect patchRect = clamp(sp.getX() - r, sp.getY() - r, 2 * r, 2 * r, cols, rows);
                // 搜索窗中等余量：够吸收局部对齐偏差(修正“有螺丝却低分”)，又不至于够到旁边的孔
                int margin = Math.max(8, (int) (r * 0.6));
                Rect searchRect = clamp(sp.getX() - r - margin, sp.getY() - r - margin,
                        2 * r + 2 * margin, 2 * r + 2 * margin, cols, rows);

                double edgeScore = 0.0;
                double grayScore = 0.0;
                if (patchRect.width > 3 && patchRect.height > 3
                        && searchRect.width > patchRect.width && searchRect.height > patchRect.height) {
                    edgeScore = matchScore(cGrad, tGrad, searchRect, patchRect);
                    grayScore = matchScore(alignedGray, templateGray, searchRect, patchRect);
                }
                // 取两路最优：边缘匹配抗光照，灰度匹配补充
                double score = Math.max(edgeScore, grayScore);

                log.info("  螺丝 {} 边缘={} 灰度={} 取={} (阈值<{}={}判漏打)",
                        sp.getId(), fmt(edgeScore), fmt(grayScore), fmt(score),
                        spec.getScrewMinScore(), score < spec.getScrewMinScore() ? "是" : "否");

                if (score < spec.getScrewMinScore()) {
                    defects.add(Defect.builder()
                            .type("SCREW_MISSING")
                            .shape("CIRCLE")
                            .x(sp.getX())
                            .y(sp.getY())
                            .r(r + 4)
                            .message(String.format("螺丝 %s 漏打/异常 (边缘%.2f 灰度%.2f)",
                                    sp.getId() == null ? "?" : sp.getId(), edgeScore, grayScore))
                            .build());
                }
            }
            return defects;
        } finally {
            tGrad.release();
            cGrad.release();
        }
    }

    /** 裁出 search/patch 子图后调用 EdgeUtil 匹配，返回最大归一化相关得分。 */
    private static double matchScore(Mat searchImg, Mat patchImg, Rect searchRect, Rect patchRect) {
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
