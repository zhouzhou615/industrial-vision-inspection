package com.vision.inspect.detect;

import com.vision.inspect.model.Defect;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * 在（已对齐的）采图上圈出缺陷：漏打螺丝画红圈，Logo 问题画红框，并标注编号。
 */
public final class DefectAnnotator {

    private static final Scalar RED = new Scalar(0, 0, 255);   // BGR
    private static final Scalar GREEN = new Scalar(0, 200, 0);

    private DefectAnnotator() {
    }

    /**
     * @param alignedColor 已对齐采图（BGR），本方法在其副本上绘制
     * @param defects      缺陷列表
     * @return 标注后的新图（调用方负责 release）
     */
    public static Mat annotate(Mat alignedColor, List<Defect> defects) {
        Mat out = alignedColor.clone();
        int thickness = Math.max(2, out.cols() / 500);
        int idx = 1;
        for (Defect d : defects) {
            if ("CIRCLE".equals(d.getShape())) {
                Imgproc.circle(out, new Point(d.getX(), d.getY()), Math.max(6, d.getR()), RED, thickness);
                putLabel(out, String.valueOf(idx), d.getX() + d.getR(), d.getY() - d.getR(), thickness);
            } else { // RECT
                Imgproc.rectangle(out, new Point(d.getX(), d.getY()),
                        new Point(d.getX() + d.getW(), d.getY() + d.getH()), RED, thickness);
                putLabel(out, String.valueOf(idx), d.getX(), d.getY() - 6, thickness);
            }
            idx++;
        }
        // 顶部整体结论条
        String head = defects.isEmpty() ? "OK" : ("NG  x" + defects.size());
        Imgproc.putText(out, head, new Point(20, 50), Imgproc.FONT_HERSHEY_SIMPLEX,
                1.5, defects.isEmpty() ? GREEN : RED, thickness + 1);
        return out;
    }

    private static void putLabel(Mat img, String text, int x, int y, int thickness) {
        Imgproc.putText(img, text, new Point(Math.max(0, x), Math.max(15, y)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, RED, thickness);
    }
}
