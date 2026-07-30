package com.vision.inspect.controller;

import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.service.LineResultStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * 历史图像：把产线检测存档的标注图按时间倒序列出，标明所属产品与 OK/NG。
 */
@RestController
@RequestMapping("/api/v1/history")
public class HistoryController {

    private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final VisionProperties properties;
    private final LineResultStore lineResultStore;

    public HistoryController(VisionProperties properties, LineResultStore lineResultStore) {
        this.properties = properties;
        this.lineResultStore = lineResultStore;
    }

    /**
     * 列出历史图像。
     *
     * @param product 产品/配方编码过滤（可空）
     * @param result  OK / NG / 全部（可空）
     * @param limit   返回条数上限
     */
    @GetMapping("/images")
    public List<Map<String, Object>> images(@RequestParam(required = false) String product,
                                            @RequestParam(required = false) String result,
                                            @RequestParam(defaultValue = "200") int limit) {
        // 内存中最近结果（含判定与缺陷信息），用路径做索引以便与磁盘文件对应
        Map<String, Object[]> byPath = new LinkedHashMap<>();
        for (var r : lineResultStore.snapshot().recent) {
            if (r.getAnnotatedImagePath() != null) {
                byPath.put(norm(r.getAnnotatedImagePath()),
                        new Object[]{r.isPassed(), r.getMessage(), r.getScrewMissing(), r.getScrewExpected()});
            }
        }

        List<Map<String, Object>> out = new ArrayList<>();
        Path base = Path.of(properties.getCapture().getOutputDir());
        if (!Files.isDirectory(base)) {
            return out;
        }
        try (Stream<Path> products = Files.list(base)) {
            List<Path> dirs = products.filter(Files::isDirectory).toList();
            for (Path pd : dirs) {
                String productCode = pd.getFileName().toString();
                if (product != null && !product.isBlank() && !productCode.contains(product)) {
                    continue;
                }
                Path annotated = pd.resolve("annotated");
                if (!Files.isDirectory(annotated)) {
                    continue;
                }
                try (Stream<Path> files = Files.list(annotated)) {
                    for (Path f : files.filter(Files::isRegularFile).toList()) {
                        String name = f.getFileName().toString();
                        if (!name.toLowerCase().endsWith(".jpg") && !name.toLowerCase().endsWith(".png")) {
                            continue;
                        }
                        Map<String, Object> item = new LinkedHashMap<>();
                        item.put("productCode", productCode);
                        item.put("file", name);
                        // 供前端展示的图片地址（沿用既有图片服务）
                        item.put("url", "/api/v1/image/capture?path="
                                + urlEncode(productCode + "/annotated/" + name));
                        long ms = fileTime(f);
                        item.put("time", LocalDateTime.ofInstant(Instant.ofEpochMilli(ms),
                                ZoneId.systemDefault()).format(TS));
                        item.put("ts", ms);

                        Object[] meta = byPath.get(norm(f.toString()));
                        if (meta != null) {
                            item.put("passed", meta[0]);
                            item.put("message", meta[1]);
                            item.put("screwMissing", meta[2]);
                            item.put("screwExpected", meta[3]);
                        } else {
                            item.put("passed", null);   // 历史文件（本次启动前）无判定详情
                            item.put("message", "");
                        }
                        out.add(item);
                    }
                } catch (IOException ignored) {
                    // 跳过该目录
                }
            }
        } catch (IOException ignored) {
            return out;
        }

        out.sort(Comparator.comparingLong(m -> -((Number) m.get("ts")).longValue()));
        if (result != null && !result.isBlank() && !"ALL".equalsIgnoreCase(result)) {
            boolean wantOk = "OK".equalsIgnoreCase(result);
            out.removeIf(m -> !(Boolean.valueOf(wantOk).equals(m.get("passed"))));
        }
        return out.size() > limit ? out.subList(0, limit) : out;
    }

    private long fileTime(Path f) {
        try {
            return Files.getLastModifiedTime(f).toMillis();
        } catch (IOException e) {
            return 0L;
        }
    }

    private String norm(String p) {
        return p == null ? "" : p.replace('\\', '/');
    }

    private String urlEncode(String s) {
        return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8)
                .replace("+", "%20");
    }
}
