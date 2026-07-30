package com.vision.inspect.controller;

import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.template.TemplateManager;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 为 Web HMI 提供图片读取：标准图 / 指定采图或差异图（限定在 data 目录内，防目录穿越）。
 */
@RestController
@RequestMapping("/api/v1/image")
public class ImageController {

    private final VisionProperties properties;
    private final TemplateManager templateManager;
    private final com.vision.inspect.template.InspectionSpecManager specManager;

    public ImageController(VisionProperties properties, TemplateManager templateManager,
                           com.vision.inspect.template.InspectionSpecManager specManager) {
        this.properties = properties;
        this.templateManager = templateManager;
        this.specManager = specManager;
    }

    /** 读取某产品的标准图 */
    @GetMapping("/template/{productCode}")
    public ResponseEntity<Resource> template(@PathVariable String productCode) {
        Path path = templateManager.getTemplateImagePath(productCode);
        return serve(path);
    }

    /**
     * 读取采图目录下的任意图片（采图/差异图）。
     * relativePath 相对于 capture.output-dir，例如 SKU-001/capture_xxx.jpg
     */
    @GetMapping("/capture")
    public ResponseEntity<Resource> capture(@RequestParam("path") String relativePath) {
        Path base = Path.of(properties.getCapture().getOutputDir()).toAbsolutePath().normalize();
        Path target = base.resolve(relativePath).normalize();
        if (!target.startsWith(base)) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "非法路径");
        }
        return serve(target);
    }

    /**
     * 标准图（带示教标注）：在标准图上画出该配方设置的螺丝位(青圈+编号)与 Logo 区(黄框)，
     * 便于在「视觉检测」页直观看到本配方要检测哪些点位。
     */
    @GetMapping(value = "/template/{productCode}/marked", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<byte[]> templateMarked(@PathVariable String productCode) {
        java.nio.file.Path tpl = templateManager.getTemplateImagePath(productCode);
        if (!Files.exists(tpl)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "标准图不存在");
        }
        org.opencv.core.Mat img = com.vision.inspect.detect.ImageIoUtil.read(tpl);
        if (img.empty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "标准图无法读取");
        }
        try {
            com.vision.inspect.model.InspectionSpec spec =
                    specManager.load(productCode).orElseGet(com.vision.inspect.model.InspectionSpec::new);
            org.opencv.core.Mat marked =
                    com.vision.inspect.detect.DefectAnnotator.annotate(img, java.util.List.of(), spec);
            org.opencv.core.MatOfByte buf = new org.opencv.core.MatOfByte();
            org.opencv.imgcodecs.Imgcodecs.imencode(".jpg", marked, buf);
            byte[] bytes = buf.toArray();
            buf.release();
            marked.release();
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .cacheControl(org.springframework.http.CacheControl.noCache())
                    .body(bytes);
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "标注标准图失败");
        } finally {
            img.release();
        }
    }

    /**
     * 实时预览：代理相机采图服务的最新一帧，供“检测画面/图像采集”页实时显示。
     */
    @GetMapping(value = "/live", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<byte[]> live() {
        try {
            String url = properties.getCamera().getBridgeUrl() + "/grab";
            java.net.HttpURLConnection conn =
                    (java.net.HttpURLConnection) java.net.URI.create(url).toURL().openConnection();
            conn.setConnectTimeout(1500);
            conn.setReadTimeout(4000);
            if (conn.getResponseCode() != 200) {
                throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "采图服务无图像");
            }
            byte[] jpg;
            try (java.io.InputStream in = conn.getInputStream()) {
                jpg = in.readAllBytes();
            }
            conn.disconnect();
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .cacheControl(org.springframework.http.CacheControl.noCache())
                    .body(jpg);
        } catch (ResponseStatusException e) {
            throw e;
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "采图服务未连接");
        }
    }

    private ResponseEntity<Resource> serve(Path path) {
        if (!Files.exists(path) || !Files.isRegularFile(path)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "图片不存在");
        }
        MediaType type = MediaType.IMAGE_JPEG;
        try {
            String probe = Files.probeContentType(path);
            if (probe != null) {
                type = MediaType.parseMediaType(probe);
            }
        } catch (IOException ignored) {
            // 使用默认 jpeg
        }
        return ResponseEntity.ok().contentType(type).body(new FileSystemResource(path));
    }
}
