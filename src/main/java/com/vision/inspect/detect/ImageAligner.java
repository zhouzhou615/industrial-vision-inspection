package com.vision.inspect.detect;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

/**
 * 把采图校正到标准图坐标系，使标准图上示教的螺丝/Logo 框“随工件位置变化”而不是固定在画面。
 *
 * <p>主方法：ORB 特征 + 部分仿射（旋转+平移+等比缩放）估计 —— <b>旋转不变，能处理任意角度</b>，
 * 这就是海康 VisionMaster “特征匹配定位 + 位置修正” 的原理。</p>
 * <p>退化：ORB 失败 → ECC 灰度配准（仅小角度）；再失败 → 仅缩放。</p>
 */
public final class ImageAligner {

    private ImageAligner() {
    }

    public static Mat align(Mat templateColor, Mat capturedColor) {
        Size size = templateColor.size();
        // 采图先缩放到标准图尺寸，统一坐标系
        Mat capResized = new Mat();
        Imgproc.resize(capturedColor, capResized, size);

        Mat tGray = enhance(templateColor);
        Mat cGray = enhance(capResized);
        try {
            // 1. ORB 仿射对齐（抗大角度旋转）
            Mat M = tryOrbAffine(tGray, cGray);
            if (M != null) {
                Mat warped = new Mat();
                Imgproc.warpAffine(capResized, warped, M, size);  // M: 采图→标准图
                M.release();
                return warped;
            }
            // 2. ECC 精配准（小角度）
            Mat warp = tryEcc(tGray, cGray);
            if (warp != null) {
                Mat warped = new Mat();
                Imgproc.warpAffine(capResized, warped, warp, size,
                        Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);
                warp.release();
                return warped;
            }
            return capResized.clone(); // 兜底：仅缩放
        } catch (Exception e) {
            return capResized.clone();
        } finally {
            tGray.release();
            cGray.release();
            capResized.release();
        }
    }

    /** 灰度化 + CLAHE 局部对比度增强（改善暗图/不均匀光照下的特征）。 */
    private static Mat enhance(Mat color) {
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY);
        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        clahe.apply(gray, gray);
        return gray;
    }

    /**
     * ORB + 部分仿射估计。tGray/cGray 同尺寸（采图已缩放到标准图尺寸）。
     * 为提速在缩小图上提特征，平移量按比例还原到全分辨率。
     * @return 2x3 仿射矩阵，映射 采图坐标 → 标准图坐标；失败返回 null。
     */
    private static Mat tryOrbAffine(Mat tGray, Mat cGray) {
        double k = Math.min(1.0, 1000.0 / tGray.cols());
        Mat tS = new Mat();
        Mat cS = new Mat();
        if (k < 1.0) {
            Imgproc.resize(tGray, tS, new Size(), k, k, Imgproc.INTER_AREA);
            Imgproc.resize(cGray, cS, new Size(), k, k, Imgproc.INTER_AREA);
        } else {
            tS = tGray.clone();
            cS = cGray.clone();
        }
        Mat desT = new Mat();
        Mat desC = new Mat();
        MatOfKeyPoint kpT = new MatOfKeyPoint();
        MatOfKeyPoint kpC = new MatOfKeyPoint();
        try {
            ORB orb = ORB.create(4000);
            orb.detectAndCompute(tS, new Mat(), kpT, desT);
            orb.detectAndCompute(cS, new Mat(), kpC, desC);
            if (desT.empty() || desC.empty()) {
                return null;
            }
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            List<MatOfDMatch> knn = new ArrayList<>();
            matcher.knnMatch(desT, desC, knn, 2); // query=模板, train=采图
            List<DMatch> good = new ArrayList<>();
            for (MatOfDMatch mm : knn) {
                DMatch[] arr = mm.toArray();
                if (arr.length >= 2 && arr[0].distance < 0.80f * arr[1].distance) {
                    good.add(arr[0]);
                }
            }
            if (good.size() < 12) {
                return null;
            }
            KeyPoint[] tk = kpT.toArray();
            KeyPoint[] ck = kpC.toArray();
            List<Point> from = new ArrayList<>(); // 采图点
            List<Point> to = new ArrayList<>();   // 标准图点
            for (DMatch m : good) {
                to.add(tk[m.queryIdx].pt);
                from.add(ck[m.trainIdx].pt);
            }
            MatOfPoint2f fromPts = new MatOfPoint2f(from.toArray(new Point[0]));
            MatOfPoint2f toPts = new MatOfPoint2f(to.toArray(new Point[0]));
            Mat inliers = new Mat();
            Mat Md = Calib3d.estimateAffinePartial2D(fromPts, toPts, inliers, Calib3d.RANSAC, 5.0, 2000, 0.99, 10);
            fromPts.release();
            toPts.release();
            inliers.release();
            if (Md == null || Md.empty() || Md.rows() != 2) {
                return null;
            }
            // 缩放/旋转合理性检查
            double scale = Math.hypot(Md.get(0, 0)[0], Md.get(1, 0)[0]);
            if (scale < 0.3 || scale > 3.0) {
                Md.release();
                return null;
            }
            // 平移量还原到全分辨率（旋转/缩放部分与尺度无关）
            if (k < 1.0) {
                Md.put(0, 2, Md.get(0, 2)[0] / k);
                Md.put(1, 2, Md.get(1, 2)[0] / k);
            }
            return Md;
        } catch (Exception e) {
            return null;
        } finally {
            tS.release();
            cS.release();
            desT.release();
            desC.release();
        }
    }

    /** ECC 灰度配准（欧氏：旋转+平移），仅适合小角度。失败返回 null。 */
    private static Mat tryEcc(Mat tGray, Mat cGray) {
        try {
            double procScale = Math.min(1.0, 1024.0 / tGray.cols());
            Mat tSmall = new Mat();
            Mat cSmall = new Mat();
            if (procScale < 1.0) {
                Imgproc.resize(tGray, tSmall, new Size(), procScale, procScale, Imgproc.INTER_AREA);
                Imgproc.resize(cGray, cSmall, new Size(), procScale, procScale, Imgproc.INTER_AREA);
            } else {
                tSmall = tGray.clone();
                cSmall = cGray.clone();
            }
            Mat warp = Mat.eye(2, 3, CvType.CV_32F);
            TermCriteria crit = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 100, 1e-5);
            double cc = Video.findTransformECC(tSmall, cSmall, warp, Video.MOTION_EUCLIDEAN, crit, new Mat(), 5);
            tSmall.release();
            cSmall.release();
            if (procScale < 1.0) {
                warp.put(0, 2, warp.get(0, 2)[0] / procScale);
                warp.put(1, 2, warp.get(1, 2)[0] / procScale);
            }
            if (Double.isNaN(cc) || cc < 0.3) {
                warp.release();
                return null;
            }
            return warp;
        } catch (Exception e) {
            return null;
        }
    }
}
