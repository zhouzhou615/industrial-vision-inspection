package com.vision.inspect.service;

import com.vision.inspect.camera.CameraFactory;
import com.vision.inspect.camera.IndustrialCamera;
import com.vision.inspect.camera.MockIndustrialCamera;
import com.vision.inspect.compare.CompareScore;
import com.vision.inspect.compare.ImageComparator;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.InspectResult;
import com.vision.inspect.model.RoiRegion;
import com.vision.inspect.signal.SignalOutput;
import com.vision.inspect.template.TemplateManager;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Optional;

/**
 * 检测业务流程编排：加载标准图 → 相机采图 → 比对 → 阈值判定 → 输出信号 → 存档。
 */
@Service
public class VisionInspectService {

    private static final Logger log = LoggerFactory.getLogger(VisionInspectService.class);
    private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS");

    private final CameraFactory cameraFactory;
    private final TemplateManager templateManager;
    private final ImageComparator imageComparator;
    private final VisionProperties properties;
    private final SignalOutput signalOutput;

    public VisionInspectService(CameraFactory cameraFactory,
                                TemplateManager templateManager,
                                ImageComparator imageComparator,
                                VisionProperties properties,
                                SignalOutput signalOutput) {
        this.cameraFactory = cameraFactory;
        this.templateManager = templateManager;
        this.imageComparator = imageComparator;
        this.properties = properties;
        this.signalOutput = signalOutput;
    }

    /**
     * 触发一次检测：相机采图并与标准图比对。
     */
    public InspectResult inspect(String productCode) {
        long start = System.currentTimeMillis();
        if (!templateManager.templateExists(productCode)) {
            throw new IllegalArgumentException("产品未注册标准图: " + productCode);
        }

        IndustrialCamera camera = cameraFactory.getActiveCamera();
        if (!camera.isOpened()) {
            camera.open();
            camera.setExposure(properties.getCamera().getExposureUs());
            camera.setGain(properties.getCamera().getGain());
        }
        camera.softwareTrigger();
        Mat captured = grabWithRetry(camera);

        Mat template = com.vision.inspect.detect.ImageIoUtil.read(templateManager.getTemplateImagePath(productCode).toString());
        try {
            Optional<RoiRegion> roi = properties.getCompare().isEnableRoi()
                    ? safeLoadRoi(productCode)
                    : Optional.empty();

            CompareScore score = imageComparator.compare(template, captured, roi);
            double threshold = properties.getCompare().getSimilarityThreshold();
            boolean passed = score.getSimilarity() >= threshold;

            Path capturePath = saveCapture(productCode, captured);
            Path diffPath = saveDiff(productCode, score);

            if (passed) {
                signalOutput.outputOk();
            } else {
                signalOutput.outputNg();
            }

            InspectResult result = InspectResult.builder()
                    .productCode(productCode)
                    .passed(passed)
                    .similarity(round4(score.getSimilarity()))
                    .threshold(threshold)
                    .algorithm(score.getAlgorithm())
                    .message(passed ? "检测通过" : "检测不合格: " + score.getDetail())
                    .templatePath(templateManager.getTemplateImagePath(productCode).toString())
                    .capturePath(capturePath.toString())
                    .diffImagePath(diffPath != null ? diffPath.toString() : null)
                    .inspectTime(LocalDateTime.now())
                    .elapsedMs(System.currentTimeMillis() - start)
                    .build();
            log.info("检测完成 product={} passed={} similarity={} elapsedMs={}",
                    productCode, passed, result.getSimilarity(), result.getElapsedMs());
            return result;
        } finally {
            captured.release();
            template.release();
        }
    }

    /**
     * 上传待测图进行检测（无相机联调）。
     */
    public InspectResult inspectFromFile(String productCode, Path imagePath) {
        IndustrialCamera camera = cameraFactory.getActiveCamera();
        if (camera instanceof MockIndustrialCamera mock) {
            mock.setMockImagePath(imagePath);
            return inspect(productCode);
        }
        // 非 mock 相机：直接读取文件做比对，不触发硬件
        return inspectStatic(productCode, imagePath);
    }

    /**
     * 直接对给定图片文件做比对（不经过相机），用于离线调试。
     */
    private InspectResult inspectStatic(String productCode, Path imagePath) {
        long start = System.currentTimeMillis();
        if (!templateManager.templateExists(productCode)) {
            throw new IllegalArgumentException("产品未注册标准图: " + productCode);
        }
        Mat captured = com.vision.inspect.detect.ImageIoUtil.read(imagePath.toString());
        if (captured.empty()) {
            throw new IllegalArgumentException("无法读取待测图片");
        }
        Mat template = com.vision.inspect.detect.ImageIoUtil.read(templateManager.getTemplateImagePath(productCode).toString());
        try {
            Optional<RoiRegion> roi = properties.getCompare().isEnableRoi()
                    ? safeLoadRoi(productCode)
                    : Optional.empty();
            CompareScore score = imageComparator.compare(template, captured, roi);
            double threshold = properties.getCompare().getSimilarityThreshold();
            boolean passed = score.getSimilarity() >= threshold;
            Path capturePath = saveCapture(productCode, captured);
            Path diffPath = saveDiff(productCode, score);
            if (passed) {
                signalOutput.outputOk();
            } else {
                signalOutput.outputNg();
            }
            return InspectResult.builder()
                    .productCode(productCode)
                    .passed(passed)
                    .similarity(round4(score.getSimilarity()))
                    .threshold(threshold)
                    .algorithm(score.getAlgorithm())
                    .message(passed ? "检测通过" : "检测不合格: " + score.getDetail())
                    .templatePath(templateManager.getTemplateImagePath(productCode).toString())
                    .capturePath(capturePath.toString())
                    .diffImagePath(diffPath != null ? diffPath.toString() : null)
                    .inspectTime(LocalDateTime.now())
                    .elapsedMs(System.currentTimeMillis() - start)
                    .build();
        } finally {
            captured.release();
            template.release();
        }
    }

    /**
     * 采图失败重试 1 次。
     */
    private Mat grabWithRetry(IndustrialCamera camera) {
        try {
            Mat frame = camera.grabFrame();
            if (frame != null && !frame.empty()) {
                return frame;
            }
        } catch (RuntimeException first) {
            log.warn("采图失败，重试一次: {}", first.getMessage());
        }
        Mat retry = camera.grabFrame();
        if (retry == null || retry.empty()) {
            throw new IllegalStateException("采图失败（已重试）");
        }
        return retry;
    }

    private Optional<RoiRegion> safeLoadRoi(String productCode) {
        try {
            return templateManager.loadRoi(productCode);
        } catch (Exception e) {
            log.warn("读取 ROI 失败，忽略 ROI: {}", e.getMessage());
            return Optional.empty();
        }
    }

    private Path saveCapture(String productCode, Mat captured) {
        try {
            Path dir = Path.of(properties.getCapture().getOutputDir(), productCode);
            Files.createDirectories(dir);
            String ts = LocalDateTime.now().format(TS);
            Path path = dir.resolve("capture_" + ts + ".jpg");
            com.vision.inspect.detect.ImageIoUtil.write(path, captured);
            return path;
        } catch (Exception e) {
            throw new IllegalStateException("保存采图失败", e);
        }
    }

    private Path saveDiff(String productCode, CompareScore score) {
        if (score.getDiffImage() == null || score.getDiffImage().empty()) {
            return null;
        }
        try {
            Path dir = Path.of(properties.getCapture().getOutputDir(), productCode, "diff");
            Files.createDirectories(dir);
            String ts = LocalDateTime.now().format(TS);
            Path path = dir.resolve("diff_" + ts + ".jpg");
            com.vision.inspect.detect.ImageIoUtil.write(path, score.getDiffImage());
            score.getDiffImage().release();
            return path;
        } catch (Exception e) {
            log.warn("保存差异图失败: {}", e.getMessage());
            return null;
        }
    }

    private double round4(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}
