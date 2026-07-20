package com.vision.inspect.config;

import jakarta.annotation.PostConstruct;
import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Configuration;

/**
 * 启动时加载 OpenCV 本地库（由 org.openpnp:opencv 自带，无需手动安装）。
 */
@Configuration
public class OpenCvConfig {

    private static final Logger log = LoggerFactory.getLogger(OpenCvConfig.class);

    @PostConstruct
    public void loadNativeLibrary() {
        OpenCV.loadLocally();
        log.info("OpenCV loaded: {}", Core.VERSION);
    }
}
