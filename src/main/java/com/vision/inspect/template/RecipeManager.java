package com.vision.inspect.template;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vision.inspect.config.VisionProperties;
import com.vision.inspect.model.Recipe;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * 工程配方管理：持久化到 data/recipes.json。
 * 一个产品可有多个配方；同一时刻只有一个配方处于“启用”状态，产线按它检测。
 */
@Component
public class RecipeManager {

    private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final VisionProperties properties;
    private final ObjectMapper mapper = new ObjectMapper();

    public RecipeManager(VisionProperties properties) {
        this.properties = properties;
    }

    private Path storePath() {
        return Path.of(properties.getTemplate().getBaseDir()).getParent() == null
                ? Path.of("data", "recipes.json")
                : Path.of(properties.getTemplate().getBaseDir()).getParent().resolve("recipes.json");
    }

    public synchronized List<Recipe> list() {
        Path p = storePath();
        if (!Files.exists(p)) {
            return new ArrayList<>();
        }
        try {
            return mapper.readValue(p.toFile(), new TypeReference<List<Recipe>>() {
            });
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    private synchronized void saveAll(List<Recipe> list) throws IOException {
        Path p = storePath();
        Files.createDirectories(p.getParent());
        mapper.writerWithDefaultPrettyPrinter().writeValue(p.toFile(), list);
    }

    /** 新增或更新配方（按 code 唯一）。 */
    public synchronized Recipe save(Recipe r) throws IOException {
        if (r.getCode() == null || r.getCode().isBlank()) {
            // 配方编码 = 产品编码@配方名，作为存储目录名
            String name = r.getName() == null || r.getName().isBlank() ? "默认" : r.getName();
            r.setCode(r.getProductCode() + "@" + name);
        }
        String now = LocalDateTime.now().format(TS);
        List<Recipe> list = list();
        Optional<Recipe> old = list.stream().filter(x -> x.getCode().equals(r.getCode())).findFirst();
        if (old.isPresent()) {
            Recipe o = old.get();
            r.setCreateTime(o.getCreateTime());
            r.setCreateUser(o.getCreateUser());
            r.setEnabled(o.isEnabled());
            list.remove(o);
        } else {
            r.setCreateTime(now);
            if (r.getCreateUser() == null) {
                r.setCreateUser("admin");
            }
        }
        r.setUpdateTime(now);
        if (r.getUpdateUser() == null) {
            r.setUpdateUser("admin");
        }
        list.add(r);
        saveAll(list);
        return r;
    }

    /** 启用指定配方（其余自动停用）。 */
    public synchronized void enable(String code) throws IOException {
        List<Recipe> list = list();
        for (Recipe r : list) {
            r.setEnabled(r.getCode().equals(code));
        }
        saveAll(list);
    }

    public synchronized void delete(String code) throws IOException {
        List<Recipe> list = list();
        list.removeIf(r -> r.getCode().equals(code));
        saveAll(list);
    }

    /** 当前启用的配方编码；没有则返回空。 */
    public Optional<String> activeCode() {
        return list().stream().filter(Recipe::isEnabled).map(Recipe::getCode).findFirst();
    }
}
