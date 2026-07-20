package com.vision.inspect.template;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.InspectionSpec;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * 工件检测配置（螺丝位/Logo 区）的持久化：templates/{code}/spec.json。
 */
@Component
public class InspectionSpecManager {

    private final VisionProperties properties;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public InspectionSpecManager(VisionProperties properties) {
        this.properties = properties;
    }

    private Path specPath(String productCode) {
        return Path.of(properties.getTemplate().getBaseDir(), productCode, "spec.json");
    }

    public void save(String productCode, InspectionSpec spec) throws IOException {
        Path dir = Path.of(properties.getTemplate().getBaseDir(), productCode);
        Files.createDirectories(dir);
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(specPath(productCode).toFile(), spec);
    }

    public Optional<InspectionSpec> load(String productCode) throws IOException {
        Path p = specPath(productCode);
        if (!Files.exists(p)) {
            return Optional.empty();
        }
        return Optional.of(objectMapper.readValue(p.toFile(), InspectionSpec.class));
    }

    public boolean exists(String productCode) {
        return Files.exists(specPath(productCode));
    }
}
