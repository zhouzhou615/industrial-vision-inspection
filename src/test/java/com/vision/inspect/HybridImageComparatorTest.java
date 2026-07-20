package com.vision.inspect;

import com.vision.inspect.compare.CompareScore;
import com.vision.inspect.compare.HybridImageComparator;
import com.vision.inspect.config.VisionProperties;
import nu.pattern.OpenCV;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 比对算法基本性质验证：相同图相似度高，差异大图相似度低。
 */
class HybridImageComparatorTest {

    @BeforeAll
    static void loadOpenCv() {
        OpenCV.loadLocally();
    }

    @Test
    void identicalImagesHaveHighSimilarity() {
        VisionProperties props = new VisionProperties();
        HybridImageComparator cmp = new HybridImageComparator(props);

        Mat a = new Mat(200, 200, CvType.CV_8UC3, new Scalar(120, 120, 120));
        // 画一个白色方块作为特征，便于模板匹配
        Imgproc.rectangle(a, new Point(60, 60), new Point(140, 140),
                new Scalar(255, 255, 255), -1);
        Mat b = a.clone();

        CompareScore score = cmp.compare(a, b, Optional.empty());
        assertTrue(score.getSimilarity() > 0.9,
                "相同图相似度应 > 0.9，实际=" + score.getSimilarity());
        a.release();
        b.release();
    }

    @Test
    void differentImagesHaveLowerSimilarity() {
        VisionProperties props = new VisionProperties();
        HybridImageComparator cmp = new HybridImageComparator(props);

        Mat a = new Mat(200, 200, CvType.CV_8UC3, new Scalar(20, 20, 20));
        Mat b = new Mat(200, 200, CvType.CV_8UC3, new Scalar(220, 220, 220));

        CompareScore identical = cmp.compare(a, a.clone(), Optional.empty());
        CompareScore different = cmp.compare(a, b, Optional.empty());

        assertTrue(different.getSimilarity() < identical.getSimilarity(),
                "差异大的图相似度应更低");
        a.release();
        b.release();
    }
}
