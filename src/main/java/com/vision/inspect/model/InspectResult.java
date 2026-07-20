package com.vision.inspect.model;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 单次检测结果，作为 REST 响应体并用于追溯存档。
 * 后半部分字段用于工件检测（螺丝/Logo）模式，整图相似度模式下为空。
 */
@Data
@Builder
public class InspectResult {
    private String productCode;
    private boolean passed;
    private double similarity;
    private double threshold;
    private String algorithm;
    private String message;
    private String templatePath;
    private String capturePath;
    private String diffImagePath;
    private LocalDateTime inspectTime;
    private long elapsedMs;

    // ==== 工件检测（螺丝/Logo）扩展字段 ====
    /** 缺陷列表 */
    private List<Defect> defects;
    /** 标注了缺陷圈选的结果图路径 */
    private String annotatedImagePath;
    /** 螺丝总数 / 漏打数 */
    private Integer screwExpected;
    private Integer screwMissing;
    /** Logo 是否合格 / 歪斜角度（度） */
    private Boolean logoPassed;
    private Double logoSkewDeg;
    /** 是否已触发报警 */
    private Boolean alarmTriggered;
}
