package com.vision.inspect.camera;

import com.vision.inspect.config.VisionProperties;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 模拟相机：从样例图读取，便于无硬件联调。
 * 默认启用（matchIfMissing = true）。
 */
@Component
@ConditionalOnProperty(name = "vision.camera.provider", havingValue = "mock", matchIfMissing = true)
public class MockIndustrialCamera implements IndustrialCamera {

    private final VisionProperties properties;
    private boolean opened;
    private Path mockImagePath;

    public MockIndustrialCamera(VisionProperties properties) {
        this.properties = properties;
    }

    public void setMockImagePath(Path mockImagePath) {
        this.mockImagePath = mockImagePath;
    }

    @Override
    public void open() {
        opened = true;
    }

    @Override
    public void close() {
        opened = false;
    }

    @Override
    public boolean isOpened() {
        return opened;
    }

    @Override
    public Mat grabFrame() {
        if (!opened) {
            throw new IllegalStateException("相机未打开");
        }
        if (mockImagePath != null && Files.exists(mockImagePath)) {
            return Imgcodecs.imread(mockImagePath.toString());
        }
        Path fallback = Path.of(properties.getCapture().getOutputDir(), "last_capture.jpg");
        if (Files.exists(fallback)) {
            return Imgcodecs.imread(fallback.toString());
        }
        throw new IllegalStateException("Mock 相机未配置图像，请设置 mockImagePath 或放置 last_capture.jpg");
    }

    @Override
    public void setExposure(int exposureUs) {
        // mock 忽略
    }

    @Override
    public void setGain(double gain) {
        // mock 忽略
    }

    @Override
    public void softwareTrigger() {
        // mock 忽略
    }
}
