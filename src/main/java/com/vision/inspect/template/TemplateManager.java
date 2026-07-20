package com.vision.inspect.template;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.RoiRegion;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * 标准图（模板）管理：每个产品编码对应一张基准图及可选 ROI 配置。
 */
@Component
public class TemplateManager {

    private final VisionProperties properties;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public TemplateManager(VisionProperties properties) {
        this.properties = properties;
    }

    public Path getTemplateImagePath(String productCode) {
        return Path.of(properties.getTemplate().getBaseDir(), productCode, "template.jpg");
    }

    public Path getRoiConfigPath(String productCode) {
        return Path.of(properties.getTemplate().getBaseDir(), productCode, "roi.json");
    }

    public void saveTemplate(String productCode, byte[] imageBytes) throws IOException {
        Path dir = Path.of(properties.getTemplate().getBaseDir(), productCode);
        Files.createDirectories(dir);
        Files.write(dir.resolve("template.jpg"), imageBytes);
    }

    public void saveRoi(String productCode, RoiRegion roi) throws IOException {
        Path dir = Path.of(properties.getTemplate().getBaseDir(), productCode);
        Files.createDirectories(dir);
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(dir.resolve("roi.json").toFile(), roi);
    }

    public Optional<RoiRegion> loadRoi(String productCode) throws IOException {
        Path roiPath = getRoiConfigPath(productCode);
        if (!Files.exists(roiPath)) {
            return Optional.empty();
        }
        return Optional.of(objectMapper.readValue(roiPath.toFile(), RoiRegion.class));
    }

    public boolean templateExists(String productCode) {
        return Files.exists(getTemplateImagePath(productCode));
    }

    /**
     * 列出所有已注册产品编码（即 templates 目录下含 template.jpg 的子目录）。
     */
    public List<String> listProductCodes() {
        Path base = Path.of(properties.getTemplate().getBaseDir());
        if (!Files.isDirectory(base)) {
            return List.of();
        }
        List<String> codes = new ArrayList<>();
        try (Stream<Path> children = Files.list(base)) {
            children.filter(Files::isDirectory)
                    .filter(p -> Files.exists(p.resolve("template.jpg")))
                    .map(p -> p.getFileName().toString())
                    .sorted(Comparator.naturalOrder())
                    .forEach(codes::add);
        } catch (IOException e) {
            // 忽略，返回已收集部分
        }
        return codes;
    }

    @Data
    public static class TemplateMeta {
        private String productCode;
        private String description;
        private double defaultThreshold;
    }
}
