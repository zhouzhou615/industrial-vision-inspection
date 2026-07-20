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

    public ImageController(VisionProperties properties, TemplateManager templateManager) {
        this.properties = properties;
        this.templateManager = templateManager;
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
