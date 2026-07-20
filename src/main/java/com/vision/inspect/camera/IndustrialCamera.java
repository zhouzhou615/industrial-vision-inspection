package com.vision.inspect.camera;

import org.opencv.core.Mat;

/**
 * 工业相机统一接口。实际产线可对接海康 MVS、Basler Pylon、大华等 SDK。
 */
public interface IndustrialCamera extends AutoCloseable {

    void open();

    @Override
    void close();

    boolean isOpened();

    Mat grabFrame();

    void setExposure(int exposureUs);

    void setGain(double gain);

    void softwareTrigger();
}
