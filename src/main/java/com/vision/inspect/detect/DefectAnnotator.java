package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.LogoSpec;
import com.vision.inspect.model.ScrewPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * 在（已对齐的）采图上圈出缺陷：漏打螺丝画红圈，Logo 问题画红框，并标注编号。
 * 同时用淡色画出“正在检测的区域”（螺丝位=青圈，Logo 框=黄框），便于确认改动是否落在检测区内。
 */
public final class DefectAnnotator {

    private static final Scalar RED = new Scalar(0, 0, 255);      // BGR
    private static final Scalar GREEN = new Scalar(0, 200, 0);
    private static final Scalar CYAN = new Scalar(220, 220, 0);   // 螺丝检测区
    private static final Scalar YELLOW = new Scalar(0, 220, 220); // Logo 检测区

    private DefectAnnotator() {
    }

    /** 先画出所有被检测的区域（淡色），再叠加缺陷标注。 */
    public static Mat annotate(Mat alignedColor, List<Defect> defects, InspectionSpec spec) {
        Mat out = alignedColor.clone();
        int thickness = Math.max(2, out.cols() / 700);
        if (spec != null) {
            if (spec.getScrews() != null) {
                for (ScrewPoint s : spec.getScrews()) {
                    Imgproc.circle(out, new Point(s.getX(), s.getY()), Math.max(6, s.getR()), CYAN, 1);
                }
            }
            if (spec.getLogos() != null) {
                int li = 1;
                for (LogoSpec lg : spec.getLogos()) {
                    Imgproc.rectangle(out, new Point(lg.getX(), lg.getY()),
                            new Point(lg.getX() + lg.getWidth(), lg.getY() + lg.getHeight()), YELLOW, 1);
                    putLabel(out, "L" + li, lg.getX(), lg.getY() - 4, 1, YELLOW);
                    li++;
                }
            }
        }
        drawDefects(out, defects, thickness);
        return out;
    }

    /**
     * @param alignedColor 已对齐采图（BGR），本方法在其副本上绘制
     * @param defects      缺陷列表
     * @return 标注后的新图（调用方负责 release）
     */
    public static Mat annotate(Mat alignedColor, List<Defect> defects) {
        Mat out = alignedColor.clone();
        drawDefects(out, defects, Math.max(2, out.cols() / 500));
        return out;
    }

    private static void drawDefects(Mat out, List<Defect> defects, int thickness) {
        int idx = 1;
        for (Defect d : defects) {
            if ("CIRCLE".equals(d.getShape())) {
                Imgproc.circle(out, new Point(d.getX(), d.getY()), Math.max(6, d.getR()), RED, thickness);
                putLabel(out, String.valueOf(idx), d.getX() + d.getR(), d.getY() - d.getR(), thickness, RED);
            } else { // RECT
                Imgproc.rectangle(out, new Point(d.getX(), d.getY()),
                        new Point(d.getX() + d.getW(), d.getY() + d.getH()), RED, thickness);
                putLabel(out, String.valueOf(idx), d.getX(), d.getY() - 6, thickness, RED);
            }
            idx++;
        }
        String head = defects.isEmpty() ? "OK" : ("NG  x" + defects.size());
        Imgproc.putText(out, head, new Point(20, 50), Imgproc.FONT_HERSHEY_SIMPLEX,
                1.5, defects.isEmpty() ? GREEN : RED, thickness + 1);
    }

    private static void putLabel(Mat img, String text, int x, int y, int thickness, Scalar color) {
        Imgproc.putText(img, text, new Point(Math.max(0, x), Math.max(15, y)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness);
    }
}
