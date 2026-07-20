package com.vision.inspect.camera;

import com.vision.inspect.config.VisionProperties;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * USB 相机实现（OpenCV VideoCapture），可用于开发调试。
 * 产线 GigE 相机请替换为厂商 SDK 实现类。
 */
@Component
@ConditionalOnProperty(name = "vision.camera.provider", havingValue = "opencv")
public class OpenCvUsbCamera implements IndustrialCamera {

    private final VisionProperties properties;
    private VideoCapture capture;

    public OpenCvUsbCamera(VisionProperties properties) {
        this.properties = properties;
    }

    @Override
    public void open() {
        capture = new VideoCapture(properties.getCamera().getDeviceId());
        if (!capture.isOpened()) {
            throw new IllegalStateException("无法打开 USB 相机 deviceId=" + properties.getCamera().getDeviceId());
        }
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1920);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 1080);
    }

    @Override
    public void close() {
        if (capture != null) {
            capture.release();
            capture = null;
        }
    }

    @Override
    public boolean isOpened() {
        return capture != null && capture.isOpened();
    }

    @Override
    public Mat grabFrame() {
        Mat frame = new Mat();
        if (!capture.read(frame) || frame.empty()) {
            throw new IllegalStateException("采图失败");
        }
        return frame;
    }

    @Override
    public void setExposure(int exposureUs) {
        if (capture != null) {
            capture.set(Videoio.CAP_PROP_EXPOSURE, exposureUs / 1000.0);
        }
    }

    @Override
    public void setGain(double gain) {
        if (capture != null) {
            capture.set(Videoio.CAP_PROP_GAIN, gain);
        }
    }

    @Override
    public void softwareTrigger() {
        grabFrame().release();
    }
}
