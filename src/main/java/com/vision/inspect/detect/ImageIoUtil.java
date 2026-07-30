package com.vision.inspect.detect;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 图片读写工具：绕开 OpenCV 在 Windows 下无法处理<b>非 ASCII（中文）路径</b>的限制。
 *
 * <p>{@code Imgcodecs.imread/imwrite} 直接传中文路径会失败（读到空 Mat / 写不出文件），
 * 因此这里统一用 Java NIO 读写字节流，再由 OpenCV 做编解码。</p>
 */
public final class ImageIoUtil {

    private ImageIoUtil() {
    }

    /** 读取图片为 Mat（支持中文路径）；失败返回空 Mat。 */
    public static Mat read(Path path) {
        try {
            byte[] bytes = Files.readAllBytes(path);
            MatOfByte buf = new MatOfByte(bytes);
            Mat img = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR);
            buf.release();
            return img == null ? new Mat() : img;
        } catch (IOException e) {
            return new Mat();
        }
    }

    public static Mat read(String path) {
        return read(Path.of(path));
    }

    /** 写出 JPG（支持中文路径）。 */
    public static boolean write(Path path, Mat img, int quality) {
        MatOfByte buf = new MatOfByte();
        try {
            MatOfInt params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality);
            if (!Imgcodecs.imencode(".jpg", img, buf, params)) {
                return false;
            }
            params.release();
            Files.createDirectories(path.getParent());
            Files.write(path, buf.toArray());
            return true;
        } catch (IOException e) {
            return false;
        } finally {
            buf.release();
        }
    }

    public static boolean write(Path path, Mat img) {
        return write(path, img, 92);
    }
}
