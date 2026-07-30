package com.vision.inspect.service;

import com.vision.inspect.camera.CameraFactory;
import com.vision.inspect.camera.IndustrialCamera;
import com.vision.inspect.camera.MockIndustrialCamera;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.detect.DefectAnnotator;
import com.vision.inspect.detect.ImageAligner;
import com.vision.inspect.detect.LogoInspector;
import com.vision.inspect.detect.ScrewDetector;
import com.vision.inspect.detect.ScrewPatternDetector;
import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectResult;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.LogoSpec;
import com.vision.inspect.model.ScrewPoint;
import com.vision.inspect.signal.AlarmService;
import com.vision.inspect.signal.SignalOutput;
import com.vision.inspect.template.InspectionSpecManager;
import com.vision.inspect.template.TemplateManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * 工件检测流程：螺丝漏打 + Logo 错误/歪斜检测，圈选缺陷并触发报警。
 */
@Service
public class PlateInspectService {

    private static final Logger log = LoggerFactory.getLogger(PlateInspectService.class);
    private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS");

    /**
     * 检测工作分辨率上限：大图(如 5120×5120)降到此宽度再检测，速度快约 10 倍，坐标按比例缩放。
     *
     * <p>螺丝检测已改为逐颗模板匹配（见 {@link ScrewPatternDetector}），不再依赖“暗斑候选点 +
     * 点阵投票配准”，因此<b>不再有“分辨率高了候选点暴涨导致配准歧义”的问题</b>，这个值可以调大。
     * 唯一的约束是标准图与采图必须同源同分辨率（采图端 {@code hik_camera_server --max-width}
     * 决定采图宽度）；不一致时下面会自动统一并告警。</p>
     */
    private static final int WORK_MAX_WIDTH = 1600;

    private final CameraFactory cameraFactory;
    private final TemplateManager templateManager;
    private final InspectionSpecManager specManager;
    private final VisionProperties properties;
    private final SignalOutput signalOutput;
    private final AlarmService alarmService;

    public PlateInspectService(CameraFactory cameraFactory,
                               TemplateManager templateManager,
                               InspectionSpecManager specManager,
                               VisionProperties properties,
                               SignalOutput signalOutput,
                               AlarmService alarmService) {
        this.cameraFactory = cameraFactory;
        this.templateManager = templateManager;
        this.specManager = specManager;
        this.properties = properties;
        this.signalOutput = signalOutput;
        this.alarmService = alarmService;
    }

    /** 相机采图 → 工件检测 */
    public InspectResult inspect(String productCode) {
        IndustrialCamera camera = cameraFactory.getActiveCamera();
        if (!camera.isOpened()) {
            camera.open();
            camera.setExposure(properties.getCamera().getExposureUs());
            camera.setGain(properties.getCamera().getGain());
        }
        camera.softwareTrigger();
        Mat captured = camera.grabFrame();
        return runInspection(productCode, captured);
    }

    /** 上传图片 → 工件检测（离线调试） */
    public InspectResult inspectFromFile(String productCode, Path imagePath) {
        IndustrialCamera camera = cameraFactory.getActiveCamera();
        if (camera instanceof MockIndustrialCamera) {
            Mat captured = com.vision.inspect.detect.ImageIoUtil.read(imagePath.toString());
            if (captured.empty()) {
                throw new IllegalArgumentException("无法读取待测图片");
            }
            return runInspection(productCode, captured);
        }
        Mat captured = com.vision.inspect.detect.ImageIoUtil.read(imagePath.toString());
        if (captured.empty()) {
            throw new IllegalArgumentException("无法读取待测图片");
        }
        return runInspection(productCode, captured);
    }

    private InspectResult runInspection(String productCode, Mat captured) {
        long start = System.currentTimeMillis();
        if (!templateManager.templateExists(productCode)) {
            captured.release();
            throw new IllegalArgumentException("产品未注册标准图: " + productCode);
        }
        // 检测配置可选：没配置就用空配置（不强制标螺丝/Logo，没标的就不检测）
        InspectionSpec spec;
        try {
            spec = specManager.load(productCode).orElseGet(InspectionSpec::new);
        } catch (java.io.IOException e) {
            captured.release();
            throw new IllegalStateException("读取检测配置失败", e);
        }

        Mat template = com.vision.inspect.detect.ImageIoUtil.read(templateManager.getTemplateImagePath(productCode).toString());
        Mat aligned = null;
        Mat templateGray = new Mat();
        Mat alignedGray = new Mat();
        Mat annotated = null;
        try {
            // 先把采图统一到标准图尺寸。采图端(python hik_camera_server --max-width)可能已经缩过图，
            // 此时采图分辨率与标准图不同，而下面的 scale 是按「标准图」算的：若直接套到采图上等于缩了两次。
            // 实测 5120 标准图 + 1200 采图 → 采图被缩到 375 再由 ImageAligner 拉回 1600，
            // 相当于 4 倍模糊放大，螺丝候选点与图案配准全部失效（判定每次都不同）。
            if (captured.cols() != template.cols() || captured.rows() != template.rows()) {
                log.warn("采图 {}x{} 与标准图 {}x{} 尺寸不一致，先统一到标准图尺寸（建议重新注册标准图，使两者同源同分辨率）",
                        captured.cols(), captured.rows(), template.cols(), template.rows());
                Mat cFit = new Mat();
                Imgproc.resize(captured, cFit, template.size(), 0, 0,
                        captured.cols() > template.cols() ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
                captured.release();
                captured = cFit;
            }

            // 降采样到工作分辨率：大图检测太慢(5120² ~3s)，缩到 1600 宽约快 10 倍，坐标同比缩放，结果等价
            if (template.cols() > WORK_MAX_WIDTH) {
                double scale = (double) WORK_MAX_WIDTH / template.cols();
                Mat tSmall = new Mat();
                Imgproc.resize(template, tSmall, new Size(), scale, scale, Imgproc.INTER_AREA);
                template.release();
                template = tSmall;
                Mat cSmall = new Mat();
                Imgproc.resize(captured, cSmall, new Size(), scale, scale, Imgproc.INTER_AREA);
                captured.release();
                captured = cSmall;
                spec = scaleSpec(spec, scale);
            }
            // 剔除落在图像之外的示教点（旧分辨率残留），否则会把图案配准整体带偏
            spec = dropOutOfBoundsScrews(spec, template.cols(), template.rows());

            aligned = ImageAligner.align(template, captured);
            Imgproc.cvtColor(template, templateGray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(aligned, alignedGray, Imgproc.COLOR_BGR2GRAY);

            List<Defect> defects = new ArrayList<>();

            // 1. 螺丝漏打（逐颗模板匹配：用螺丝自身邻域定位，不依赖整图对齐，可精确指出哪一颗）
            //    注意：这里必须传<b>原始灰度</b>，不能先做 equalizeHist。全局直方图均衡是按整帧
            //    直方图做的重映射，而标准图与采图露出的暗背景多少不同 → 两边被映射成不同的灰度，
            //    小 patch 的归一化相关随之失真。实测均衡后「有螺丝最低分 0.66 / 空孔最高分 0.65」
            //    几乎无法区分，改用原始灰度后变成「0.88 / 0.72」，间隔从 0.02 拉到 0.15。
            ScrewPatternDetector.Result screwResult =
                    ScrewPatternDetector.detect(templateGray, alignedGray, spec);
            List<Defect> screwDefects = screwResult.defects;
            defects.addAll(screwDefects);
            int screwExpected = spec.getScrews().size();
            int screwMissing = screwDefects.size();

            // 2. Logo 错误/歪斜（支持多个 Logo 位；没标就跳过）
            //    Logo 区域大得多，全局均衡对它是净收益（抵消整体明暗差异），故在此处才做。
            Boolean logoPassed = null;
            Double logoSkew = null;
            if (spec.getLogos() != null && !spec.getLogos().isEmpty()) {
                Imgproc.equalizeHist(templateGray, templateGray);
                Imgproc.equalizeHist(alignedGray, alignedGray);
                logoPassed = true;
                double worstSkew = 0.0;
                int idx = 1;
                for (com.vision.inspect.model.LogoSpec lg : spec.getLogos()) {
                    LogoInspector.LogoResult lr = LogoInspector.inspect(templateGray, alignedGray, lg);
                    log.info("  Logo{} 匹配得分={} 最大变化块={}%/{}像素 (阈值 得分<{} 或 变化>{}% 或 >40像素)",
                            idx, lr.score, Math.round(lr.diffRatio * 100), lr.blobPixels,
                            lg.getMinScore(), Math.round(lg.getMaxDiffRatio() * 100));
                    if (Math.abs(lr.skewDeg) > Math.abs(worstSkew)) {
                        worstSkew = lr.skewDeg;
                    }
                    if (!lr.passed) {
                        logoPassed = false;
                        if (lr.defect != null) {
                            lr.defect.setMessage("Logo" + idx + ": " + lr.defect.getMessage());
                            defects.add(lr.defect);
                        }
                    }
                    idx++;
                }
                logoSkew = worstSkew;
            }

            boolean passed = defects.isEmpty();

            // 3. 标注缺陷图（同时画出被检测的区域，便于确认改动是否落在检测区）
            annotated = DefectAnnotator.annotate(aligned, defects, spec,
                    screwResult.dx, screwResult.dy);
            Path capturePath = saveImage(productCode, "capture", captured);
            Path annotatedPath = saveImage(productCode, "annotated", annotated);

            InspectResult result = InspectResult.builder()
                    .productCode(productCode)
                    .passed(passed)
                    .algorithm("plate-screw-logo")
                    .message(passed ? "检测通过" : "检出 " + defects.size() + " 处缺陷")
                    .templatePath(templateManager.getTemplateImagePath(productCode).toString())
                    .capturePath(capturePath.toString())
                    .annotatedImagePath(annotatedPath.toString())
                    .defects(defects)
                    .screwExpected(screwExpected)
                    .screwMissing(screwMissing)
                    .logoPassed(logoPassed)
                    .logoSkewDeg(logoSkew)
                    .inspectTime(LocalDateTime.now())
                    .elapsedMs(System.currentTimeMillis() - start)
                    .build();

            // 4. 信号 / 报警
            if (passed) {
                signalOutput.outputOk();
                result.setAlarmTriggered(false);
            } else {
                alarmService.raise(result);
                result.setAlarmTriggered(true);
            }
            log.info("工件检测 product={} passed={} screwMissing={}/{} logoPassed={} elapsedMs={}",
                    productCode, passed, screwMissing, screwExpected, logoPassed, result.getElapsedMs());
            return result;
        } finally {
            captured.release();
            template.release();
            templateGray.release();
            alignedGray.release();
            if (aligned != null) aligned.release();
            if (annotated != null) annotated.release();
        }
    }

    /**
     * 剔除落在检测图之外的示教螺丝点。
     *
     * <p>标准图被重新注册成另一种分辨率后（例如从 5120 换成采图端 1200），示教页载入的是旧坐标，
     * 用户在新图上补点后一起保存，spec 里就混着两套坐标系。这些旧点永远匹配不上，还会
     * <b>把图案配准的投票带偏</b>——实测 6 个残留点让 dx/dy 每次检测都不同，好件也会被判 NG。
     * 这里直接丢弃并明确告警，提示重新示教。</p>
     */
    private InspectionSpec dropOutOfBoundsScrews(InspectionSpec spec, int cols, int rows) {
        if (spec.getScrews() == null || spec.getScrews().isEmpty()) {
            return spec;
        }
        List<ScrewPoint> keep = new ArrayList<>();
        List<String> dropped = new ArrayList<>();
        for (ScrewPoint p : spec.getScrews()) {
            if (p.getX() >= 0 && p.getX() < cols && p.getY() >= 0 && p.getY() < rows) {
                keep.add(p);
            } else {
                dropped.add(String.format("%s(%d,%d)", p.getId(), p.getX(), p.getY()));
            }
        }
        if (dropped.isEmpty()) {
            return spec;
        }
        log.error("示教坐标越界，已忽略 {} 个点: {} —— 检测图仅 {}x{}。"
                        + "多半是标准图换过分辨率后旧示教点没清掉，请在「图像采集」页点「清空螺丝位」重新示教。",
                dropped.size(), dropped, cols, rows);
        InspectionSpec out = new InspectionSpec();
        out.setScrewMinScore(spec.getScrewMinScore());
        out.setScrews(keep);
        out.setLogos(spec.getLogos());
        return out;
    }

    /** 按比例缩放检测配置中的坐标（不改动持久化的原始 spec）。 */
    private InspectionSpec scaleSpec(InspectionSpec spec, double s) {
        InspectionSpec out = new InspectionSpec();
        out.setScrewMinScore(spec.getScrewMinScore());
        List<ScrewPoint> screws = new ArrayList<>();
        if (spec.getScrews() != null) {
            for (ScrewPoint p : spec.getScrews()) {
                screws.add(new ScrewPoint(p.getId(),
                        (int) Math.round(p.getX() * s), (int) Math.round(p.getY() * s),
                        Math.max(3, (int) Math.round(p.getR() * s))));
            }
        }
        out.setScrews(screws);
        List<LogoSpec> logos = new ArrayList<>();
        if (spec.getLogos() != null) {
            for (LogoSpec lg : spec.getLogos()) {
                logos.add(new LogoSpec(
                        (int) Math.round(lg.getX() * s), (int) Math.round(lg.getY() * s),
                        (int) Math.round(lg.getWidth() * s), (int) Math.round(lg.getHeight() * s),
                        lg.getMaxSkewDeg(), lg.getMinGoodMatches(), lg.getMinScore(), lg.getMaxDiffRatio()));
            }
        }
        out.setLogos(logos);
        return out;
    }

    private Path saveImage(String productCode, String sub, Mat img) {
        try {
            Path dir = "capture".equals(sub)
                    ? Path.of(properties.getCapture().getOutputDir(), productCode)
                    : Path.of(properties.getCapture().getOutputDir(), productCode, sub);
            Files.createDirectories(dir);
            String ts = LocalDateTime.now().format(TS);
            Path path = dir.resolve(sub + "_" + ts + ".jpg");
            com.vision.inspect.detect.ImageIoUtil.write(path, img);
            return path;
        } catch (Exception e) {
            throw new IllegalStateException("保存图片失败", e);
        }
    }
}
