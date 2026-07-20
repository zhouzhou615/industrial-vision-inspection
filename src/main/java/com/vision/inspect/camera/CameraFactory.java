package com.vision.inspect.camera;

import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 相机工厂：根据已启用的实现 Bean 返回当前可用相机。
 * 由 {@code @ConditionalOnProperty} 保证只有一种 provider 被注入。
 */
@Component
public class CameraFactory {

    private final List<IndustrialCamera> cameras;

    public CameraFactory(List<IndustrialCamera> cameras) {
        this.cameras = cameras;
    }

    public IndustrialCamera getActiveCamera() {
        if (cameras.isEmpty()) {
            throw new IllegalStateException("未找到可用相机实现，请检查 vision.camera.provider 配置");
        }
        if (cameras.size() > 1) {
            throw new IllegalStateException("存在多个相机 Bean，请确保仅启用一种 provider");
        }
        return cameras.get(0);
    }
}
