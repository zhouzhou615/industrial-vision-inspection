package com.vision.inspect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 标准图上一个螺丝位的示教点。
 * (x, y) 为螺丝中心（标准图像素坐标），r 为检测半径（覆盖螺丝头）。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ScrewPoint {
    private String id;   // 螺丝编号，如 S1、S2
    private int x;
    private int y;
    private int r;       // 检测半径（像素）
}
