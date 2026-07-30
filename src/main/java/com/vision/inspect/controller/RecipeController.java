package com.vision.inspect.controller;

import com.vision.inspect.model.Recipe;
import com.vision.inspect.template.RecipeManager;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 工程配方接口：一个产品可有多个配方，启用其中一个供产线检测。
 */
@RestController
@RequestMapping("/api/v1/recipe")
public class RecipeController {

    private final RecipeManager recipeManager;

    public RecipeController(RecipeManager recipeManager) {
        this.recipeManager = recipeManager;
    }

    /** 列出配方，可按产品编码过滤 */
    @GetMapping("/list")
    public List<Recipe> list(@RequestParam(required = false) String productCode) {
        List<Recipe> all = recipeManager.list();
        if (productCode == null || productCode.isBlank()) {
            return all;
        }
        return all.stream()
                .filter(r -> r.getProductCode() != null && r.getProductCode().contains(productCode))
                .collect(Collectors.toList());
    }

    /** 全部产品编码（下拉用） */
    @GetMapping("/products")
    public List<String> products() {
        return recipeManager.list().stream()
                .map(Recipe::getProductCode)
                .filter(p -> p != null && !p.isBlank())
                .distinct()
                .sorted()
                .collect(Collectors.toList());
    }

    /** 新增/更新配方 */
    @PostMapping("/save")
    public Recipe save(@RequestBody Recipe recipe) throws Exception {
        return recipeManager.save(recipe);
    }

    /** 启用配方（其余自动停用） */
    @PostMapping("/{code}/enable")
    public Map<String, Object> enable(@PathVariable String code) throws Exception {
        recipeManager.enable(code);
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("success", true);
        r.put("message", "已启用配方 " + code);
        return r;
    }

    /** 当前启用的配方 */
    @GetMapping("/active")
    public Map<String, Object> active() {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("code", recipeManager.activeCode().orElse(null));
        return r;
    }

    @DeleteMapping("/{code}")
    public Map<String, Object> delete(@PathVariable String code) throws Exception {
        recipeManager.delete(code);
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("success", true);
        return r;
    }
}
