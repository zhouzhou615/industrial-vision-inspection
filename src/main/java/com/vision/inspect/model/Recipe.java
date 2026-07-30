package com.vision.inspect.model;

import lombok.Data;

/**
 * 工程配方：一个产品可以有多个配方（如电视机的“背板螺丝”“边框螺丝”各一套检测方案）。
 *
 * <p>{@link #code} 是配方的唯一标识，同时作为标准图/检测配置的存储目录名
 * （data/templates/{code}/），因此与既有检测流程完全兼容。</p>
 */
@Data
public class Recipe {
    /** 配方唯一编码（存储目录名），如 TV-001@背板 */
    private String code;
    /** 产品编码，如 TV-001 */
    private String productCode;
    /** 配方名称，如 背板螺丝 */
    private String name;
    /** 版本号 */
    private String version = "V1.0";
    /** 是否启用（当前产线运行使用的配方） */
    private boolean enabled;
    private String createTime;
    private String createUser;
    private String updateTime;
    private String updateUser;
    /** 备注 */
    private String remark;
}
