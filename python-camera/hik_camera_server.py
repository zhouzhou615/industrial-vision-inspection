# -*- coding: utf-8 -*-
"""
海康 MV-CH250-90GM 流水线采图服务（连续采集 + 传感器 LineStatus 分件）。

工作方式（传感器接相机 Line0）：
  相机连续采集 -> 后台线程逐帧读取，并轮询 LineStatus 判断检测区有无工件：
    · 检测到工件（有料）  -> 开始缓存该件的帧，实时算清晰度(拉普拉斯方差)，只保留最清晰的一帧
    · 工件离开（无料）    -> 把最清晰那一帧 POST 给 Java 做螺丝/Logo 检测（每件只发 1 张）
  Java 端做检测 + 缺陷圈选 + 报警 + 记入看板。

同时提供 HTTP 便于手动调试：
  GET /health      -> ok
  GET /grab        -> 返回当前最新一帧 JPEG
  GET /status      -> 传感器/统计 JSON

运行前准备：
  1. MVS 4.6.3 已安装且 BasicDemo 能出图；传感器接相机 Line0。
  2. pip install numpy opencv-python
  3. python hik_camera_server.py --product SKU-001 --java-url http://127.0.0.1:8088
     可选: --index 0 --exposure 10000 --gain 1.0 --line Line0 --active-high 1
           --min-frames 2 --port 5000
"""
import argparse
import ctypes
import json
import os
import sys
import threading
import time
from ctypes import c_bool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import urllib.request

# ==== 让 Python 找到 MVS SDK 的核心运行库 MvCameraControl.dll ====
# Python 3.8+ 不再自动搜 PATH，必须显式把 Runtime 目录加入 DLL 搜索路径。
SDK_RUNTIME_CANDIDATES = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
    r"D:\Program Files\MVS\MVS\Runtime\Win64",
]
for _rt in SDK_RUNTIME_CANDIDATES:
    if os.path.isdir(_rt):
        os.environ["PATH"] = _rt + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_rt)  # Python 3.8+ 关键：显式加 DLL 目录
        except (AttributeError, OSError):
            pass
        print("MVS 运行库目录: %s" % _rt)
        break

# ==== 让 Python 找到 MVS SDK 的 MvImport 模块 ====
SDK_MVIMPORT = r"D:\Program Files\MVS\MVS\Development\Samples\Python\MvImport"
if os.path.isdir(SDK_MVIMPORT):
    sys.path.append(SDK_MVIMPORT)

from MvCameraControl_class import *
from CameraParams_header import *
from CameraParams_const import *
from PixelType_header import *
from MvErrorDefine_const import *

import numpy as np
import cv2

cam = None
CFG = None

# 运行时共享状态
state_lock = threading.Lock()
latest_jpeg = None            # 最新一帧（供 /grab）
sensor_active = False         # 当前传感器是否有料
stats = {"items": 0, "sent": 0, "last": None}


def to_hex(num):
    if num < 0:
        num += 2 ** 32
    return hex(num)


def check(ret, action):
    if ret != MV_OK:
        raise RuntimeError("%s 失败, ret=%s" % (action, to_hex(ret)))


def open_camera(cfg):
    """枚举 -> 打开 -> 连续采集 -> 配置传感器输入线。"""
    global cam
    MvCamera.MV_CC_Initialize()

    device_list = MV_CC_DEVICE_INFO_LIST()
    check(MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list), "枚举设备")
    if device_list.nDeviceNum == 0:
        raise RuntimeError("未找到 GigE 相机")
    index = cfg.index if cfg.index < device_list.nDeviceNum else 0
    print("找到 %d 台相机, 使用索引 %d" % (device_list.nDeviceNum, index))

    st_dev = ctypes.cast(device_list.pDeviceInfo[index],
                         ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    check(cam.MV_CC_CreateHandle(st_dev), "创建句柄")
    check(cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "打开设备")

    packet = cam.MV_CC_GetOptimalPacketSize()
    if packet > 0:
        cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet)
        print("最优包大小: %d" % packet)

    # 连续采集（关闭触发），避免软/硬触发的状态问题
    check(cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF), "关闭触发(连续采集)")

    # 固定曝光/增益
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetEnumValue("GainAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", float(cfg.exposure))
    cam.MV_CC_SetFloatValue("Gain", float(cfg.gain))

    # 配置补光频闪输出：相机每次曝光时在该输出线打脉冲驱动光源，实现采集时自动补光
    if cfg.strobe:
        try:
            cam.MV_CC_SetEnumValueByString("LineSelector", cfg.strobe_line)
            cam.MV_CC_SetEnumValueByString("LineMode", "Strobe")
            cam.MV_CC_SetBoolValue("StrobeEnable", True)
            if cfg.strobe_duration >= 0:
                # 0 = 跟随曝光时长；>0 = 固定脉宽(微秒)
                cam.MV_CC_SetIntValue("StrobeLineDuration", int(cfg.strobe_duration))
            if cfg.strobe_delay != 0:
                cam.MV_CC_SetIntValue("StrobeLineDelay", int(cfg.strobe_delay))
            print("已启用补光频闪输出: %s (曝光时自动打光)" % cfg.strobe_line)
        except Exception as e:
            print("警告: 配置补光频闪失败(%s)，请在 MVS 的 属性→数字IO 里手动确认 %s 设为 Strobe 并开启 StrobeEnable"
                  % (e, cfg.strobe_line))

    # 最后选择传感器输入线，保证运行期读取的 LineStatus 是传感器那条线
    try:
        cam.MV_CC_SetEnumValueByString("LineSelector", cfg.line)
    except Exception as e:
        print("警告: 设置 LineSelector=%s 失败(%s)，仍尝试读取 LineStatus" % (cfg.line, e))

    check(cam.MV_CC_StartGrabbing(), "开始取流")
    print("相机已就绪（连续采集），传感器线: %s，工件产品号: %s" % (cfg.line, cfg.product))


def read_line_status():
    """读取传感器电平，按极性返回是否有料。读取失败返回 None。"""
    b = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("LineStatus", b)
    if ret != MV_OK:
        return None
    level = bool(b.value)
    return level if CFG.active_high else (not level)


def frame_to_gray(st_frame):
    """MV_FRAME_OUT -> numpy 灰度图（Mono8 单通道相机）。"""
    w = st_frame.stFrameInfo.nWidth
    h = st_frame.stFrameInfo.nHeight
    n = st_frame.stFrameInfo.nFrameLen
    buf = ctypes.string_at(st_frame.pBufAddr, n)
    # Mono8 与其它单通道格式都按灰度 reshape
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w)


def sharpness(gray):
    """拉普拉斯方差衡量清晰度；大图先缩小加速。"""
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25) if gray.shape[1] > 1600 else gray
    return cv2.Laplacian(small, cv2.CV_64F).var()


def post_frame_to_java(jpg_bytes, frame_count):
    """把最清晰帧以 multipart 上传到 Java 检测接口。"""
    url = "%s/api/v1/inspect/%s/plate/frame" % (CFG.java_url.rstrip("/"), CFG.product)
    boundary = "----visionboundary%d" % int(time.time() * 1000)
    parts = []
    parts.append(("--" + boundary).encode())
    parts.append(('Content-Disposition: form-data; name="image"; filename="best.jpg"').encode())
    parts.append(b"Content-Type: image/jpeg")
    parts.append(b"")
    parts.append(jpg_bytes)
    parts.append(("--" + boundary + "--").encode())
    body = b"\r\n".join(parts)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        passed = data.get("passed")
        with state_lock:
            stats["sent"] += 1
            stats["last"] = {"passed": passed, "message": data.get("message"),
                             "screwMissing": data.get("screwMissing"),
                             "screwExpected": data.get("screwExpected"),
                             "frames": frame_count}
        print("[件#%d] 已检测 -> %s  %s (该件采 %d 帧)" %
              (stats["items"], "OK" if passed else "NG", data.get("message"), frame_count))
    except Exception as e:
        print("上传 Java 检测失败: %s" % e)


def capture_loop():
    """连续取帧 + LineStatus 状态机分件。"""
    global latest_jpeg, sensor_active
    present = False
    best_gray = None
    best_sharp = -1.0
    frame_count = 0
    active_streak = 0      # 连续有料计数（进入防抖）
    inactive_streak = 0    # 连续无料计数（离开防抖）

    while True:
        st_frame = MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(st_frame), 0, ctypes.sizeof(st_frame))
        ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
        if ret != MV_OK:
            time.sleep(0.005)
            continue
        try:
            gray = frame_to_gray(st_frame)
            # 更新 /grab 用的最新帧（低频编码即可，这里每帧都编码方便调试）
            ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with state_lock:
                    latest_jpeg = enc.tobytes()

            active = read_line_status()
            if active is None:
                active = True  # 读不到传感器时，退化为“一直有料”（可用 /grab 手动）
            with state_lock:
                sensor_active = active

            # 连续计数做防抖
            if active:
                active_streak += 1
                inactive_streak = 0
            else:
                inactive_streak += 1
                active_streak = 0

            if not present:
                # 只有连续确认有料 enter_confirm 帧，才认定“新工件进入”，滤掉信号毛刺
                if active_streak >= CFG.enter_confirm:
                    present = True
                    best_gray = None
                    best_sharp = -1.0
                    frame_count = 0
                    with state_lock:
                        stats["items"] += 1
                    print("[件#%d] 工件进入检测区，开始采集…" % stats["items"])
            else:
                # 工件在检测区内：有料就累计最清晰帧（短暂掉料不结束）
                if active:
                    frame_count += 1
                    s = sharpness(gray)
                    if s > best_sharp:
                        best_sharp = s
                        best_gray = gray.copy()
                # 连续确认无料 leave_confirm 帧，才认定“工件离开”
                if inactive_streak >= CFG.leave_confirm:
                    present = False
                    if frame_count >= CFG.min_frames and best_gray is not None:
                        okj, encj = cv2.imencode(".jpg", best_gray,
                                                 [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        if okj:
                            jpg = encj.tobytes()
                            fc = frame_count
                            threading.Thread(target=post_frame_to_java,
                                             args=(jpg, fc), daemon=True).start()
                    elif frame_count > 0:
                        print("[件#%d] 采集帧数不足(%d)，丢弃" % (stats["items"], frame_count))
        finally:
            cam.MV_CC_FreeImageBuffer(st_frame)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._text(200, "ok")
        if path == "/status":
            with state_lock:
                snap = {"sensorActive": sensor_active, "items": stats["items"],
                        "sent": stats["sent"], "last": stats["last"]}
            return self._json(200, snap)
        if path == "/grab":
            with state_lock:
                jpg = latest_jpeg
            if jpg is None:
                return self._text(503, "尚无图像")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
        self._text(404, "not found")

    def _text(self, code, text):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--exposure", type=float, default=10000)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--product", type=str, required=True, help="产品编码，须与 Java 已注册标准图/示教一致")
    ap.add_argument("--java-url", type=str, default="http://127.0.0.1:8088")
    ap.add_argument("--line", type=str, default="Line0", help="传感器输入线，如 Line0/Line1")
    ap.add_argument("--active-high", type=int, default=1, help="1=高电平有料，0=低电平有料")
    ap.add_argument("--strobe", type=int, default=1, help="1=启用补光频闪输出(曝光时打光)，0=不控制光源")
    ap.add_argument("--strobe-line", type=str, default="Line1", help="补光灯所接的相机输出线，如 Line1/Line2")
    ap.add_argument("--strobe-duration", type=int, default=0, help="频闪脉宽(微秒)，0=跟随曝光时长")
    ap.add_argument("--strobe-delay", type=int, default=0, help="频闪延迟(微秒)")
    ap.add_argument("--min-frames", type=int, default=3, help="一件至少采集多少帧才判定")
    ap.add_argument("--enter-confirm", type=int, default=3, help="连续多少帧有料才确认工件进入(防抖，去信号毛刺)")
    ap.add_argument("--leave-confirm", type=int, default=4, help="连续多少帧无料才确认工件离开(防抖)")
    CFG = ap.parse_args()
    CFG.active_high = bool(CFG.active_high)

    open_camera(CFG)
    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", CFG.port), Handler)
        print("采图服务已启动: http://127.0.0.1:%d  (/health /grab /status)" % CFG.port)
        server.serve_forever()
    finally:
        if cam is not None:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            MvCamera.MV_CC_Finalize()
            print("相机已释放")


if __name__ == "__main__":
    main()
