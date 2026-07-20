package com.vision.inspect.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 一个缺陷标注。shape=CIRCLE 时用 (x,y)=圆心、r=半径；shape=RECT 时用 (x,y,w,h)。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Defect {
    /** SCREW_MISSING | LOGO_WRONG | LOGO_SKEW */
    private String type;
    /** CIRCLE | RECT */
    private String shape;
    private int x;
    private int y;
    private int w;
    private int h;
    private int r;
    private String message;
}
