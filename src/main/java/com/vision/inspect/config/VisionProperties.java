package com.vision.inspect.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 统一配置项，对应 application.yml 中的 vision.* 配置。
 */
@Data
@Component
@ConfigurationProperties(prefix = "vision")
public class VisionProperties {

    private Camera camera = new Camera();
    private Template template = new Template();
    private Capture capture = new Capture();
    private Compare compare = new Compare();
    private Signal signal = new Signal();
    private Vm vm = new Vm();

    @Data
    public static class Camera {
        private String provider = "mock";
        private int deviceId = 0;
        private int exposureUs = 10000;
        private double gain = 1.0;
        private String triggerMode = "software";
        /** 海康 Python 采图 sidecar 的地址（provider=hikvision 时使用） */
        private String bridgeUrl = "http://127.0.0.1:5000";
    }

    @Data
    public static class Template {
        private String baseDir = "./data/templates";
    }

    @Data
    public static class Capture {
        private String outputDir = "./data/captures";
    }

    @Data
    public static class Compare {
        private String algorithm = "hybrid";
        private double similarityThreshold = 0.92;
        private int maxOffsetPixels = 30;
        private boolean enableRoi = true;
    }

    @Data
    public static class Signal {
        private boolean enabled = true;
        private int okGpioPin = 1;
        private int ngGpioPin = 2;
    }

    /** 对接海康 VisionMaster：Java 作为 TCP 服务端接收 VM 推送的检测结果 */
    @Data
    public static class Vm {
        private boolean enabled = false;
        private int tcpPort = 9000;
        /** VM 输出的标注图所在目录（用于把图拷进采图目录供网页展示，可空） */
        private String imageDir = "";
    }
}
