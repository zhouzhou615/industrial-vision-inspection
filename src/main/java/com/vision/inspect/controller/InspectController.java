package com.vision.inspect.controller;

import com.vision.inspect.model.InspectResult;
import com.vision.inspect.model.InspectionSpec;
import com.vision.inspect.model.RoiRegion;
import com.vision.inspect.service.LineResultStore;
import com.vision.inspect.service.PlateInspectService;
import com.vision.inspect.service.VisionInspectService;
import com.vision.inspect.template.InspectionSpecManager;
import com.vision.inspect.template.TemplateManager;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * 对外 REST 接口。Base URL: http://localhost:8088
 */
@Validated
@RestController
@RequestMapping("/api/v1/inspect")
public class InspectController {

    private final VisionInspectService inspectService;
    private final PlateInspectService plateInspectService;
    private final TemplateManager templateManager;
    private final InspectionSpecManager specManager;
    private final LineResultStore lineResultStore;

    public InspectController(VisionInspectService inspectService,
                            PlateInspectService plateInspectService,
                            TemplateManager templateManager,
                            InspectionSpecManager specManager,
                            LineResultStore lineResultStore) {
        this.inspectService = inspectService;
        this.plateInspectService = plateInspectService;
        this.templateManager = templateManager;
        this.specManager = specManager;
        this.lineResultStore = lineResultStore;
    }

    /** 触发一次检测：相机采图并与标准图比对 */
    @PostMapping("/{productCode}")
    public InspectResult inspect(@PathVariable @NotBlank String productCode) {
        return inspectService.inspect(productCode);
    }

    /** 上传待测图进行检测（无相机联调） */
    @PostMapping(value = "/{productCode}/file", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public InspectResult inspectByFile(@PathVariable String productCode,
                                       @RequestPart("image") MultipartFile image) throws Exception {
        Path temp = Files.createTempFile("inspect_", ".jpg");
        image.transferTo(temp);
        try {
            return inspectService.inspectFromFile(productCode, temp);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    /** 注册/更新标准图 */
    @PostMapping(value = "/template/{productCode}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse registerTemplate(@PathVariable String productCode,
                                        @RequestPart("image") MultipartFile image) throws Exception {
        templateManager.saveTemplate(productCode, image.getBytes());
        return ApiResponse.ok("标准图已保存");
    }

    /** 配置 ROI 区域 */
    @PostMapping("/template/{productCode}/roi")
    public ApiResponse saveRoi(@PathVariable String productCode, @RequestBody RoiRegion roi) throws Exception {
        templateManager.saveRoi(productCode, roi);
        return ApiResponse.ok("ROI 已保存");
    }

    /** 查询模板是否存在 */
    @GetMapping("/template/{productCode}/exists")
    public ApiResponse templateExists(@PathVariable String productCode) {
        return ApiResponse.ok(templateManager.templateExists(productCode));
    }

    /** 列出全部已注册产品编码 */
    @GetMapping("/templates")
    public List<String> listTemplates() {
        return templateManager.listProductCodes();
    }

    // ==== 工件检测（螺丝/Logo）====

    /** 保存工件检测配置（螺丝位 + Logo 区，前端示教后调用） */
    @PostMapping("/template/{productCode}/spec")
    public ApiResponse saveSpec(@PathVariable String productCode, @RequestBody InspectionSpec spec) throws Exception {
        specManager.save(productCode, spec);
        return ApiResponse.ok("检测配置已保存");
    }

    /** 读取工件检测配置 */
    @GetMapping("/template/{productCode}/spec")
    public InspectionSpec getSpec(@PathVariable String productCode) throws Exception {
        return specManager.load(productCode).orElse(new InspectionSpec());
    }

    /**
     * 在标准图上自动识别螺丝位（霍夫圆），返回生成的检测配置供前端预览/微调后保存。
     * 若已存在配置，则沿用其中的 Logo 区与阈值。
     */
    @PostMapping("/template/{productCode}/spec/auto")
    public InspectionSpec autoDetectScrews(@PathVariable String productCode,
                                           @RequestParam(defaultValue = "8") int minR,
                                           @RequestParam(defaultValue = "40") int maxR) throws Exception {
        if (!templateManager.templateExists(productCode)) {
            throw new IllegalArgumentException("产品未注册标准图: " + productCode);
        }
        org.opencv.core.Mat template = org.opencv.imgcodecs.Imgcodecs.imread(
                templateManager.getTemplateImagePath(productCode).toString());
        try {
            InspectionSpec spec = specManager.load(productCode).orElse(new InspectionSpec());
            spec.setScrews(com.vision.inspect.detect.ScrewAutoDetector.detect(template, minR, maxR));
            return spec;
        } finally {
            template.release();
        }
    }

    /** 工件检测：相机采图，检测螺丝漏打 + Logo，圈选缺陷并报警 */
    @PostMapping("/{productCode}/plate")
    public InspectResult inspectPlate(@PathVariable @NotBlank String productCode) {
        return plateInspectService.inspect(productCode);
    }

    /** 工件检测：上传图片（离线调试） */
    @PostMapping(value = "/{productCode}/plate/file", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public InspectResult inspectPlateByFile(@PathVariable String productCode,
                                            @RequestPart("image") MultipartFile image) throws Exception {
        Path temp = Files.createTempFile("plate_", ".jpg");
        image.transferTo(temp);
        try {
            return plateInspectService.inspectFromFile(productCode, temp);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    // ==== 流水线（传感器触发，Python 采图服务每件推送最清晰帧到此）====

    /**
     * 流水线单件检测：由 Python 采图服务在工件离开检测区时，把该件最清晰的一帧上传到此，
     * 服务端做螺丝/Logo 检测、报警，并记入结果流供看板展示。
     */
    @PostMapping(value = "/{productCode}/plate/frame", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public InspectResult inspectLineFrame(@PathVariable String productCode,
                                          @RequestPart("image") MultipartFile image) throws Exception {
        Path temp = Files.createTempFile("frame_", ".jpg");
        image.transferTo(temp);
        try {
            InspectResult result = plateInspectService.inspectFromFile(productCode, temp);
            return lineResultStore.add(result);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    /** 看板：最近若干件结果 + 累计统计 */
    @GetMapping("/line/results")
    public LineResultStore.Snapshot lineResults() {
        return lineResultStore.snapshot();
    }

    /** 看板：清零统计 */
    @PostMapping("/line/reset")
    public ApiResponse lineReset() {
        lineResultStore.reset();
        return ApiResponse.ok("已清零");
    }

    @Data
    public static class ApiResponse {
        private boolean success;
        private Object data;
        private String message;

        public static ApiResponse ok(Object data) {
            ApiResponse r = new ApiResponse();
            r.success = true;
            r.data = data;
            return r;
        }

        public static ApiResponse ok(String message) {
            ApiResponse r = ok((Object) null);
            r.message = message;
            return r;
        }
    }
}
