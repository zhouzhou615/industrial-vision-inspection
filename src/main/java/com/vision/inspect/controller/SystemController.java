package com.vision.inspect.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.service.LineResultStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 系统状态与设置：磁盘剩余空间、相机/传感器状态、岗位信息等。
 */
@RestController
@RequestMapping("/api/v1/system")
public class SystemController {

    private final VisionProperties properties;
    private final LineResultStore lineResultStore;
    private final ObjectMapper mapper = new ObjectMapper();

    public SystemController(VisionProperties properties, LineResultStore lineResultStore) {
        this.properties = properties;
        this.lineResultStore = lineResultStore;
    }

    private Path settingsPath() {
        return Path.of(properties.getCapture().getOutputDir()).getParent() == null
                ? Path.of("data", "settings.json")
                : Path.of(properties.getCapture().getOutputDir()).getParent().resolve("settings.json");
    }

    /** 运行状态：存储空间、相机连接、传感器触发、设备运行。 */
    @GetMapping("/status")
    public Map<String, Object> status() {
        Map<String, Object> m = new LinkedHashMap<>();

        // 检测图片所存盘的剩余空间
        File dir = new File(properties.getCapture().getOutputDir());
        File probe = dir.exists() ? dir : new File(".");
        long total = probe.getTotalSpace();
        long free = probe.getUsableSpace();
        m.put("diskTotalGb", round1(total / 1024.0 / 1024 / 1024));
        m.put("diskFreeGb", round1(free / 1024.0 / 1024 / 1024));
        m.put("diskUsedPercent", total > 0 ? (int) Math.round((total - free) * 100.0 / total) : 0);
        m.put("capturePath", probe.getAbsolutePath());

        // 相机采图服务 / 传感器状态（来自 Python sidecar）
        Map<String, Object> cam = bridgeStatus();
        m.put("cameraConnected", cam != null);
        m.put("sensorActive", cam != null && Boolean.TRUE.equals(cam.get("sensorActive")));
        m.put("bridgeItems", cam == null ? 0 : cam.getOrDefault("items", 0));

        // 生产统计
        LineResultStore.Snapshot s = lineResultStore.snapshot();
        m.put("total", s.total);
        m.put("ok", s.ok);
        m.put("ng", s.ng);
        m.put("yield", s.total > 0 ? Math.round(s.ok * 1000.0 / s.total) / 10.0 : 0);
        m.put("running", cam != null);
        return m;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> bridgeStatus() {
        try {
            String url = properties.getCamera().getBridgeUrl() + "/status";
            HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
            conn.setConnectTimeout(800);
            conn.setReadTimeout(800);
            if (conn.getResponseCode() != 200) {
                return null;
            }
            try (InputStream in = conn.getInputStream()) {
                return mapper.readValue(in, Map.class);
            }
        } catch (Exception e) {
            return null;
        }
    }

    /** 读取系统设置 */
    @GetMapping("/settings")
    @SuppressWarnings("unchecked")
    public Map<String, Object> getSettings() {
        Path p = settingsPath();
        Map<String, Object> def = new LinkedHashMap<>();
        def.put("stationName", "1029");
        def.put("stationCode", "001");
        def.put("companyName", "工业视觉检测系统");
        def.put("capturePath", properties.getCapture().getOutputDir());
        def.put("templatePath", properties.getTemplate().getBaseDir());
        def.put("bridgeUrl", properties.getCamera().getBridgeUrl());
        def.put("keepDays", 30);
        def.put("diskWarnPercent", 90);
        def.put("saveOkImage", true);
        if (!Files.exists(p)) {
            return def;
        }
        try {
            Map<String, Object> saved = mapper.readValue(p.toFile(), Map.class);
            def.putAll(saved);
            return def;
        } catch (Exception e) {
            return def;
        }
    }

    /** 保存系统设置 */
    @PostMapping("/settings")
    public Map<String, Object> saveSettings(@RequestBody Map<String, Object> body) throws Exception {
        Path p = settingsPath();
        Files.createDirectories(p.getParent());
        mapper.writerWithDefaultPrettyPrinter().writeValue(p.toFile(), body);
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("success", true);
        r.put("message", "设置已保存");
        return r;
    }

    private double round1(double v) {
        return Math.round(v * 10) / 10.0;
    }
}
