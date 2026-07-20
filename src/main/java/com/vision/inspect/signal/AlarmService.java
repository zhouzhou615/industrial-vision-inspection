package com.vision.inspect.signal;

import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.awt.Toolkit;
import java.util.List;

/**
 * 报警服务：检出缺陷时触发。默认实现为
 * 日志高亮 + 蜂鸣 + 调用 {@link SignalOutput#outputNg()}（可对接 PLC/继电器/声光报警柱）。
 */
@Service
public class AlarmService {

    private static final Logger log = LoggerFactory.getLogger(AlarmService.class);

    private final SignalOutput signalOutput;
    private volatile String lastAlarm;

    public AlarmService(SignalOutput signalOutput) {
        this.signalOutput = signalOutput;
    }

    /**
     * 触发报警。
     */
    public void raise(InspectResult result) {
        List<Defect> defects = result.getDefects();
        StringBuilder sb = new StringBuilder();
        sb.append("产品 ").append(result.getProductCode()).append(" 检出 ")
                .append(defects == null ? 0 : defects.size()).append(" 处缺陷: ");
        if (defects != null) {
            for (Defect d : defects) {
                sb.append("[").append(d.getMessage()).append("] ");
            }
        }
        lastAlarm = sb.toString();
        log.error("★★★ 报警 ★★★ {}", lastAlarm);

        // 物理 NG 信号（GPIO/PLC/声光柱）
        try {
            signalOutput.outputNg();
        } catch (Exception e) {
            log.warn("信号输出失败: {}", e.getMessage());
        }
        // 本机蜂鸣（工控机有声卡时）
        try {
            Toolkit.getDefaultToolkit().beep();
        } catch (Exception ignored) {
            // 无图形环境时忽略
        }
    }

    public void clear() {
        lastAlarm = null;
        try {
            signalOutput.reset();
        } catch (Exception ignored) {
        }
    }

    public String getLastAlarm() {
        return lastAlarm;
    }
}
