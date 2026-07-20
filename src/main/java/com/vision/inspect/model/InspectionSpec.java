package com.vision.inspect.model;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * 工件检测配置（每个产品编码一份），由前端“示教”生成并保存为 spec.json。
 * 螺丝位、Logo 区都可选：没标的就不检测（“没有就不要标”）。
 */
@Data
public class InspectionSpec {
    /** 螺丝位列表（可空 = 不检测螺丝） */
    private List<ScrewPoint> screws = new ArrayList<>();
    /** Logo 区列表（可多个；空 = 不检测 Logo） */
    private List<LogoSpec> logos = new ArrayList<>();
    /** 螺丝存在判定阈值（匹配得分 0~1，低于此判为漏打） */
    private double screwMinScore = 0.55;
}
