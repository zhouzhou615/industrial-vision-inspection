package com.vision.inspect.camera;

import com.vision.inspect.config.VisionProperties;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * 海康 MV-CH250-90GM 相机接入（HTTP 桥接方式）。
 *
 * <p>本类不直接调用海康 native SDK，而是通过 HTTP 调用一个常驻的 Python 采图服务
 * （hik_camera_server.py，复用你已装好的 MVS Python SDK）。Java 端因此无需 native 库，
 * 也不受 SDK 版本差异影响。</p>
 *
 * <p>启用方式：application.yml 设 {@code vision.camera.provider: hikvision}，
 * 并配置 {@code vision.camera.bridge-url}（默认 http://127.0.0.1:5000）。</p>
 *
 * <p>协议约定（与 hik_camera_server.py 对应）：</p>
 * <ul>
 *   <li>GET  {bridgeUrl}/health           → 200 表示相机已就绪</li>
 *   <li>GET  {bridgeUrl}/grab?exposure=..&amp;gain=.. → 返回一帧 JPEG（image/jpeg）</li>
 * </ul>
 */
@Component
@ConditionalOnProperty(name = "vision.camera.provider", havingValue = "hikvision")
public class HikvisionCamera implements IndustrialCamera {

    private static final Logger log = LoggerFactory.getLogger(HikvisionCamera.class);

    private final VisionProperties properties;
    private volatile boolean opened;
    private int exposureUs;
    private double gain;

    public HikvisionCamera(VisionProperties properties) {
        this.properties = properties;
        this.exposureUs = properties.getCamera().getExposureUs();
        this.gain = properties.getCamera().getGain();
    }

    @Override
    public void open() {
        String health = properties.getCamera().getBridgeUrl() + "/health";
        try {
            HttpURLConnection conn = (HttpURLConnection) URI.create(health).toURL().openConnection();
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(3000);
            int code = conn.getResponseCode();
            conn.disconnect();
            if (code != 200) {
                throw new IllegalStateException("采图服务未就绪, HTTP " + code);
            }
            opened = true;
            log.info("海康采图服务已连接: {}", properties.getCamera().getBridgeUrl());
        } catch (Exception e) {
            throw new IllegalStateException("无法连接海康采图服务(" + health
                    + ")，请先启动 hik_camera_server.py: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        opened = false; // 相机由 Python 服务常驻管理，这里不关闭硬件
    }

    @Override
    public boolean isOpened() {
        return opened;
    }

    @Override
    public Mat grabFrame() {
        String url = String.format("%s/grab?exposure=%d&gain=%s",
                properties.getCamera().getBridgeUrl(), exposureUs, Double.toString(gain));
        try {
            HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(5000);
            int code = conn.getResponseCode();
            if (code != 200) {
                String msg = readBody(conn.getErrorStream());
                conn.disconnect();
                throw new IllegalStateException("采图失败 HTTP " + code + ": " + msg);
            }
            byte[] jpg;
            try (InputStream in = conn.getInputStream()) {
                jpg = in.readAllBytes();
            }
            conn.disconnect();
            if (jpg.length == 0) {
                throw new IllegalStateException("采图返回空数据");
            }
            Mat frame = Imgcodecs.imdecode(new MatOfByte(jpg), Imgcodecs.IMREAD_COLOR);
            if (frame.empty()) {
                throw new IllegalStateException("采图解码失败");
            }
            return frame;
        } catch (IllegalStateException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("采图请求异常: " + e.getMessage(), e);
        }
    }

    @Override
    public void setExposure(int exposureUs) {
        this.exposureUs = exposureUs; // 随下次 grab 一并下发给采图服务
    }

    @Override
    public void setGain(double gain) {
        this.gain = gain;
    }

    @Override
    public void softwareTrigger() {
        // 软触发在采图服务的 /grab 内部完成，这里无需单独动作
    }

    private String readBody(InputStream stream) {
        if (stream == null) {
            return "";
        }
        try (InputStream in = stream; ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            in.transferTo(bos);
            return bos.toString(StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "";
        }
    }
}
