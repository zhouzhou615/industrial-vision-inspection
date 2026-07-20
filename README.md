# 工业视觉比对检测系统 (Industrial Vision Inspection)

> 标准图 + 工业相机采图 + 自动比对 + OK/NG 输出，附 Web HMI 界面。
> Spring Boot 3.2 + OpenCV 4.9，单 JAR 部署。

## 1. 功能

| 编号 | 功能 | 说明 |
|------|------|------|
| FR-01 | 标准图管理 | 按产品编码保存/更新基准图，支持 ROI |
| FR-02 | 相机采图 | 软触发采图，可配置曝光、增益（mock/USB/厂商 SDK 可插拔） |
| FR-03 | 图像比对 | 混合算法（像素 40% + SSIM 35% + 模板匹配 25%） |
| FR-04 | 结果输出 | REST JSON + OK/NG 信号（日志/GPIO/PLC 可插拔） |
| FR-05 | 追溯存档 | 保存每次采图、差异图、相似度、检测时间 |
| FR-06 | 离线调试 | 上传图片文件代替相机采图；内置 Web 界面 |

## 2. 环境要求

- **JDK 17+**（当前机器检测到 JDK 11，需升级后再构建）
- **Maven 3.8+**
- Windows 工控机 / Linux
- 工业相机 + SDK（产线，调试可用 mock）

OpenCV 本地库由依赖 `org.openpnp:opencv` 自带，启动时 `OpenCV.loadLocally()` 自动加载，**无需手动安装 OpenCV**。

## 3. 构建与运行

```bash
cd industrial-vision-inspection
mvn clean package -DskipTests
java -jar target/industrial-vision-inspection-1.0.0.jar
```

启动后访问 Web HMI：**http://localhost:8088/**

## 4. 使用流程（Web 界面）

1. 输入产品编码（如 `SKU-001`），上传一张 **OK 标准件** 图片 → 「注册标准图」。
2. （可选）在右侧标准图上拖拽框选 **ROI** → 「保存 ROI」。
3. 点击 **「相机采图检测」**（mock 模式下需配置图源），或用 **「上传图片检测」** 做离线比对。
4. 界面显示 **OK/NG**、相似度、阈值、耗时，以及标准图 / 采图 / 差异图三联预览。

## 5. REST 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/inspect/template/{code}` | 注册标准图 (multipart `image`) |
| POST | `/api/v1/inspect/template/{code}/roi` | 配置 ROI (JSON) |
| GET  | `/api/v1/inspect/template/{code}/exists` | 模板是否存在 |
| GET  | `/api/v1/inspect/templates` | 列出全部产品编码 |
| POST | `/api/v1/inspect/{code}` | 相机采图检测 |
| POST | `/api/v1/inspect/{code}/file` | 上传图片检测 (multipart `image`) |
| GET  | `/api/v1/image/template/{code}` | 读取标准图 |
| GET  | `/api/v1/image/capture?path=...` | 读取采图/差异图 |

检测响应示例：

```json
{
  "productCode": "SKU-001",
  "passed": true,
  "similarity": 0.9678,
  "threshold": 0.92,
  "algorithm": "hybrid",
  "message": "检测通过",
  "capturePath": "./data/captures/SKU-001/capture_20260527_143022_123.jpg",
  "diffImagePath": "./data/captures/SKU-001/diff/diff_20260527_143022_123.jpg",
  "inspectTime": "2026-05-27T14:30:22",
  "elapsedMs": 186
}
```

## 6. 配置 (application.yml)

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `vision.camera.provider` | mock | 相机实现：mock / opencv / hikvision / basler |
| `vision.compare.algorithm` | hybrid | 比对算法 |
| `vision.compare.similarity-threshold` | 0.92 | OK 阈值 |
| `vision.compare.max-offset-pixels` | 30 | 允许位移（模板匹配对齐范围） |
| `vision.compare.enable-roi` | true | 是否启用 ROI |

## 7. 对接真实工业相机

实现 `IndustrialCamera` 接口并加 `@ConditionalOnProperty(name="vision.camera.provider", havingValue="hikvision")`：

- **海康 MVS**：`MV_CC_CreateHandle` → `StartGrabbing` → `GetOneFrameTimeout` → 转 `Mat`
- **Basler pylon**：`camera.grab()` → 转 `Mat`

将 `provider` 改为对应值即可，无需改其他代码。

## 8. 目录结构

```
industrial-vision-inspection/
├── pom.xml
├── README.md
├── 详细设计文档.md / .docx
├── src/main/java/com/vision/inspect/
│   ├── VisionInspectionApplication.java
│   ├── camera/        # 相机抽象与实现
│   ├── compare/       # 预处理与混合比对算法
│   ├── config/        # OpenCV 加载 + 配置项
│   ├── controller/    # REST + 图片服务 + 异常处理
│   ├── model/         # InspectResult / RoiRegion
│   ├── service/       # 检测流程编排
│   ├── signal/        # OK/NG 信号输出
│   └── template/      # 标准图与 ROI 管理
├── src/main/resources/
│   ├── application.yml
│   └── static/index.html   # Web HMI
└── src/test/java/...       # 算法单元测试
```

## 9. 单元测试

```bash
mvn test
```

验证：相同图相似度 > 0.9；差异大的图相似度更低。
