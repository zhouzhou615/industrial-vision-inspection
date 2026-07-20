package com.vision.inspect.signal;

import com.vision.inspect.config.VisionProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * 默认信号输出实现：仅打印日志。产线请替换为 GPIO / PLC Modbus 实现。
 */
@Component
public class LoggingSignalOutput implements SignalOutput {

    private static final Logger log = LoggerFactory.getLogger(LoggingSignalOutput.class);

    private final VisionProperties properties;

    public LoggingSignalOutput(VisionProperties properties) {
        this.properties = properties;
    }

    @Override
    public void outputOk() {
        if (properties.getSignal().isEnabled()) {
            log.info("[SIGNAL] OK -> GPIO pin {}", properties.getSignal().getOkGpioPin());
        }
    }

    @Override
    public void outputNg() {
        if (properties.getSignal().isEnabled()) {
            log.info("[SIGNAL] NG -> GPIO pin {}", properties.getSignal().getNgGpioPin());
        }
    }

    @Override
    public void reset() {
        log.debug("[SIGNAL] RESET");
    }
}
