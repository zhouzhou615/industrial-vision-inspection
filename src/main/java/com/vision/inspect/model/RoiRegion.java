package com.vision.inspect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 感兴趣区域（Region Of Interest），仅对该矩形区域做比对。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class RoiRegion {
    private int x;
    private int y;
    private int width;
    private int height;
}
