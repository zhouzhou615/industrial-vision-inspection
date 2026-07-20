package com.vision.inspect.service;

import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.Defect;
import com.vision.inspect.model.InspectResult;
import com.vision.inspect.signal.AlarmService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * 接收海康 VisionMaster 通过 TCP 推送的检测结果，解析后记入看板并按需报警。
 * VM 作为 TCP 客户端连接本服务（默认端口 9000），每件发送一行、以换行结尾。
 *
 * <p>约定报文格式（分号分隔，UTF-8，\n 结尾）：</p>
 * <pre>
 *   产品编码;结果;螺丝总数;漏打数;Logo结果;Logo角度;标注图路径;备注
 *   例(NG): SKU-001;NG;9;1;OK;0.3;D:/vm/out/12.jpg;S5漏打
 *   例(OK): SKU-001;OK;9;0;OK;0.1;;
 * </pre>
 * 结果字段接受 OK/NG、1/0、true/false；Logo结果接受 OK/NG/NA(或空)。
 */
@Service
@ConditionalOnProperty(name = "vision.vm.enabled", havingValue = "true")
public class VmResultServer {

    private static final Logger log = LoggerFactory.getLogger(VmResultServer.class);
    private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS");

    private final VisionProperties properties;
    private final LineResultStore lineResultStore;
    private final AlarmService alarmService;

    private volatile boolean running = false;
    private ServerSocket serverSocket;
    private Thread acceptThread;

    public VmResultServer(VisionProperties properties,
                          LineResultStore lineResultStore,
                          AlarmService alarmService) {
        this.properties = properties;
        this.lineResultStore = lineResultStore;
        this.alarmService = alarmService;
    }

    @PostConstruct
    public void start() {
        int port = properties.getVm().getTcpPort();
        running = true;
        acceptThread = new Thread(() -> acceptLoop(port), "vm-tcp-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();
        log.info("VisionMaster 结果接收服务已启动: TCP 端口 {}", port);
    }

    private void acceptLoop(int port) {
        try {
            serverSocket = new ServerSocket(port);
            while (running) {
                Socket socket = serverSocket.accept();
                Thread t = new Thread(() -> handleClient(socket), "vm-tcp-client");
                t.setDaemon(true);
                t.start();
            }
        } catch (Exception e) {
            if (running) {
                log.error("VM TCP 服务异常: {}", e.getMessage());
            }
        }
    }

    private void handleClient(Socket socket) {
        log.info("VisionMaster 已连接: {}", socket.getRemoteSocketAddress());
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while (running && (line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    handleLine(line);
                }
            }
        } catch (Exception e) {
            log.warn("VM 连接读取结束: {}", e.getMessage());
        }
    }

    private void handleLine(String line) {
        try {
            String[] f = line.split(";", -1);
            String product = get(f, 0, "UNKNOWN");
            boolean passed = parseBool(get(f, 1, "NG"));
            int screwTotal = parseInt(get(f, 2, "0"));
            int screwMissing = parseInt(get(f, 3, "0"));
            String logoRes = get(f, 4, "");
            Double logoAngle = parseDoubleOrNull(get(f, 5, ""));
            String imagePath = get(f, 6, "");
            String detail = get(f, 7, "");

            Boolean logoPassed = null;
            if (logoRes.equalsIgnoreCase("OK") || logoRes.equals("1") || logoRes.equalsIgnoreCase("true")) {
                logoPassed = true;
            } else if (logoRes.equalsIgnoreCase("NG") || logoRes.equals("0") || logoRes.equalsIgnoreCase("false")) {
                logoPassed = false;
            }

            List<Defect> defects = new ArrayList<>();
            if (!passed) {
                String msg = detail.isEmpty()
                        ? (screwMissing > 0 ? ("螺丝漏打 " + screwMissing + " 处") : "检出缺陷")
                        : detail;
                defects.add(Defect.builder().type("VM").shape("RECT").message(msg).build());
            }

            String annotated = copyVmImage(product, imagePath);

            InspectResult result = InspectResult.builder()
                    .productCode(product)
                    .passed(passed)
                    .algorithm("visionmaster")
                    .message(passed ? "检测通过" : (detail.isEmpty() ? "检出缺陷" : detail))
                    .annotatedImagePath(annotated)
                    .defects(defects)
                    .screwExpected(screwTotal)
                    .screwMissing(screwMissing)
                    .logoPassed(logoPassed)
                    .logoSkewDeg(logoAngle)
                    .inspectTime(LocalDateTime.now())
                    .alarmTriggered(!passed)
                    .build();

            lineResultStore.add(result);
            if (!passed) {
                alarmService.raise(result);
            }
            log.info("VM结果 product={} passed={} 漏打={}/{} logo={}",
                    product, passed, screwMissing, screwTotal, logoPassed);
        } catch (Exception e) {
            log.warn("解析 VM 报文失败: [{}] {}", line, e.getMessage());
        }
    }

    /** 把 VM 输出的标注图拷进采图目录，供网页看板通过既有接口展示。 */
    private String copyVmImage(String product, String imagePath) {
        if (imagePath == null || imagePath.isEmpty()) {
            return null;
        }
        try {
            Path src = Path.of(imagePath);
            if (!Files.exists(src)) {
                return null;
            }
            Path dir = Path.of(properties.getCapture().getOutputDir(), product, "vm");
            Files.createDirectories(dir);
            Path dst = dir.resolve("vm_" + LocalDateTime.now().format(TS) + ".jpg");
            Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
            return dst.toString();
        } catch (Exception e) {
            log.warn("拷贝 VM 标注图失败: {}", e.getMessage());
            return null;
        }
    }

    private String get(String[] a, int i, String def) {
        return (i < a.length && a[i] != null && !a[i].trim().isEmpty()) ? a[i].trim() : def;
    }

    private boolean parseBool(String s) {
        return s.equalsIgnoreCase("OK") || s.equals("1") || s.equalsIgnoreCase("true") || s.equalsIgnoreCase("pass");
    }

    private int parseInt(String s) {
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception e) {
            return 0;
        }
    }

    private Double parseDoubleOrNull(String s) {
        try {
            return s.isEmpty() ? null : Double.parseDouble(s.trim());
        } catch (Exception e) {
            return null;
        }
    }

    @PreDestroy
    public void stop() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (Exception ignored) {
        }
    }
}
