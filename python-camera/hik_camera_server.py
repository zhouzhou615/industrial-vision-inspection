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
from urllib.parse import urlparse, quote
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
ACTIVE_PRODUCT = None   # 跟随网页「工程配方」启用的配方编码

# 运行时共享状态
state_lock = threading.Lock()
post_busy = threading.Lock()  # 保证同一时刻只有一个 Java 检测请求，避免堆积超时
latest_jpeg = None            # 最新一帧（供 /grab，保持干净原图：网页要用它存标准图）
fov_jpeg = None               # 带视野状态标注的一帧（供 /fov 调试）
sensor_active = False         # 当前传感器是否有料
fov_now = "none"              # 当前工件在视野中的状态（none/partial/full）
fov_box_now = None            # 当前工件外接框 (x,y,w,h,面积占比)
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
    # 三色灯改由 USB/串口继电器模块(Modbus RTU)控制，见 relay_open()/set_tricolor()，此处不再用相机 IO

    # 最后选择传感器输入线，保证运行期读取的 LineStatus 是传感器那条线
    try:
        cam.MV_CC_SetEnumValueByString("LineSelector", cfg.line)
    except Exception as e:
        print("警告: 设置 LineSelector=%s 失败(%s)，仍尝试读取 LineStatus" % (cfg.line, e))

    check(cam.MV_CC_StartGrabbing(), "开始取流")
    print("相机已就绪（连续采集），传感器线: %s，配方: %s"
          % (cfg.line, cfg.product if cfg.product else "自动跟随网页启用的配方"))


# ==== 三色灯：中盛 ZS-8I0-R-10A 八路继电器模块（Modbus RTU / RS232/485）====
relay_ser = None


def _crc16(data):
    """Modbus RTU CRC16。"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def relay_open():
    """打开继电器串口。失败则三色灯不可用（不影响检测）。"""
    global relay_ser
    if not CFG or not CFG.light:
        return
    try:
        import serial  # pip install pyserial
        relay_ser = serial.Serial(CFG.relay_port, CFG.relay_baud, timeout=0.3)
        print("继电器串口已打开: %s @ %d (绿=out1 黄=out2 红=out3 蜂鸣=out4)"
              % (CFG.relay_port, CFG.relay_baud))
    except Exception as e:
        relay_ser = None
        print("警告: 打开继电器串口失败(%s)。三色灯不可用；如缺库请 pip install pyserial" % e)


# 厂家原始指令帧（功能码06写寄存器）：绿=out1 黄=out2 红=out3 蜂鸣=out4
RELAY_CMDS = {
    "green_on":   "01 06 00 00 00 01 48 0A",
    "green_off":  "01 06 00 00 00 00 89 CA",
    "yellow_on":  "01 06 00 01 00 01 19 CA",
    "yellow_off": "01 06 00 01 00 00 D8 0A",
    "red_on":     "01 06 00 02 00 01 E9 CA",
    "red_off":    "01 06 00 02 00 00 28 0A",
    "beep_on":    "01 06 00 04 00 01 09 CB",
    "beep_off":   "01 06 00 04 00 00 C8 0B",
    "all_off":    "01 06 00 34 00 00 C8 04",
}


def relay_send(key):
    """按名称发送厂家指令帧。"""
    if relay_ser is None:
        return
    try:
        relay_ser.write(bytes.fromhex(RELAY_CMDS[key].replace(" ", "")))
        relay_ser.read(8)  # 读掉应答，避免堆积
    except Exception as e:
        print("继电器写入失败(%s): %s" % (key, e))


def set_tricolor(passed):
    """None→运行中(黄)；True→OK(绿)；False→NG(红+蜂鸣)。互斥点亮。"""
    if not CFG or not CFG.light or relay_ser is None:
        return
    if passed is True:                 # OK：绿
        relay_send("yellow_off"); relay_send("red_off"); relay_send("beep_off")
        relay_send("green_on")
    elif passed is False:              # NG：红 + 蜂鸣
        relay_send("yellow_off"); relay_send("green_off")
        relay_send("red_on")
        if CFG.beep:
            relay_send("beep_on")
    else:                             # 运行中/待料：黄
        relay_send("green_off"); relay_send("red_off"); relay_send("beep_off")
        relay_send("yellow_on")


def light_selftest():
    """开机自检：黄→绿→红各亮1秒，最后回到黄(运行中)。确认继电器/接线是否生效。"""
    if not CFG or not CFG.light or relay_ser is None:
        return
    print("三色灯自检: 黄→绿→红…")
    relay_send("all_off")
    set_tricolor(None); time.sleep(1.0)   # 黄
    set_tricolor(True); time.sleep(1.0)   # 绿
    set_tricolor(False); time.sleep(1.0)  # 红(+蜂鸣)
    set_tricolor(None)                    # 回到黄=运行中
    print("三色灯自检结束（没看到灯亮 → 检查 COM口/波特率/接线）")


def read_line_status():
    """读取传感器电平，按极性返回是否有料。读取失败返回 None。"""
    b = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("LineStatus", b)
    if ret != MV_OK:
        return None
    level = bool(b.value)
    return level if CFG.active_high else (not level)


def frame_to_gray(st_frame):
    """
    MV_FRAME_OUT -> numpy 灰度图（Mono8 单通道相机）。

    相机原始分辨率很高（如 5120×5120），会导致工件表面细密网点被过度分辨、
    螺丝候选点暴增到近千个而使图案配准不稳定。因此在源头按 --max-width 缩小，
    标准图与采图统一为该尺寸，螺丝在图上约 7px，检测既稳又快。
    """
    w = st_frame.stFrameInfo.nWidth
    h = st_frame.stFrameInfo.nHeight
    n = st_frame.stFrameInfo.nFrameLen
    buf = ctypes.string_at(st_frame.pBufAddr, n)
    # Mono8 与其它单通道格式都按灰度 reshape
    gray = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    mw = getattr(CFG, "max_width", 0) or 0
    if mw > 0 and w > mw:
        k = mw / float(w)
        gray = cv2.resize(gray, (int(round(w * k)), int(round(h * k))),
                          interpolation=cv2.INTER_AREA)
    return gray


def sharpness(gray):
    """拉普拉斯方差衡量清晰度；大图先缩小加速。"""
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25) if gray.shape[1] > 1600 else gray
    return cv2.Laplacian(small, cv2.CV_64F).var()


# ==== 视野门控：只在工件「完整进入视野」时采图 ====
#
# 流水线上工件是运动的，进入/离开画面时都会被画面边界裁断。裁断的工件不能用于检测：
#  - 工件外轮廓被截断 → 定位基准失真
#  - 缺了一部分 → 那一部分的螺丝位根本不在图上，必然误报漏打
# 因此按「工件外接框是否四边都离画面边界有余量」来决定采不采。

FOV_NONE = "none"        # 画面里没有工件
FOV_PARTIAL = "partial"  # 工件在画面里，但贴到边界 —— 尚未完整进入，或已开始离开
FOV_FULL = "full"        # 工件完整在画面内，四边都有余量 —— 可以检测


def workpiece_bbox(gray):
    """
    找工件外接框。工件是亮区、背景暗，用 Otsu 自动阈值取最大亮区。

    为省时间在约 320px 宽的缩略图上做（判断"是否贴边"不需要高分辨率），
    再把坐标换算回传入图的坐标系。

    :return: (x, y, w, h, 面积占画面比例)，找不到轮廓时返回 None
    """
    h0, w0 = gray.shape[:2]
    k = 320.0 / w0 if w0 > 320 else 1.0
    small = (cv2.resize(gray, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
             if k < 1.0 else gray)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)   # 填掉工件内部网点/孔洞
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)    # 去掉零散噪点
    cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    sh, sw = small.shape[:2]
    ratio = cv2.contourArea(c) / float(sw * sh)
    x, y, w, h = cv2.boundingRect(c)
    inv = (1.0 / k) if k < 1.0 else 1.0
    return (int(x * inv), int(y * inv), int(w * inv), int(h * inv), ratio)


def fov_state(gray):
    """
    判断工件在视野中的状态，返回 (状态, 外接框)。

    余量 --fov-margin 取多大：至少要盖住工件边缘的检测/分割抖动，
    太小会在"刚好齐边"时反复跳 full/partial，太大则工件必须离边界很远才肯采图。
    """
    box = workpiece_bbox(gray)
    if box is None:
        return FOV_NONE, None
    x, y, w, h, ratio = box
    if ratio < CFG.fov_min_area:
        return FOV_NONE, box          # 亮区太小：视为画面里没有工件
    m = CFG.fov_margin
    ih, iw = gray.shape[:2]
    full = (x >= m and y >= m and (x + w) <= (iw - m) and (y + h) <= (ih - m))
    return (FOV_FULL if full else FOV_PARTIAL), box


def fov_overlay_jpeg(gray, state, box):
    """把视野状态和外接框画在图上，供 /fov 调试用（不污染 /grab 的干净原图）。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ih, iw = gray.shape[:2]
    m = CFG.fov_margin
    # 画出"必须留出的余量"边界：工件框碰到这条线就算贴边
    cv2.rectangle(vis, (m, m), (iw - m, ih - m), (0, 200, 255), 1)
    color = {FOV_FULL: (0, 220, 0), FOV_PARTIAL: (0, 165, 255), FOV_NONE: (128, 128, 128)}[state]
    if box is not None:
        x, y, w, h, ratio = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, "%s  area=%.3f" % (state, ratio), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(vis, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    ok, enc = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return enc.tobytes() if ok else None


def light_sync_loop():
    """
    三色灯跟随界面状态：工件进入检测区(尚无结果)=黄「进行中」，
    检测结果 OK=绿、NG=红。与网页「视觉检测」页显示保持一致。
    """
    last = "init"
    while True:
        try:
            with state_lock:
                active = sensor_active
                last_res = stats.get("last")
            if active and not last_res_is_fresh():
                want = "busy"
            elif last_res is not None:
                want = "ok" if last_res.get("passed") else "ng"
            else:
                want = "busy" if active else "idle"

            if want != last:
                if want == "ok":
                    set_tricolor(True)
                elif want == "ng":
                    set_tricolor(False)
                else:
                    set_tricolor(None)      # 黄=进行中/待料
                last = want
        except Exception as e:
            print("三色灯同步异常: %s" % e)
        time.sleep(0.5)


def last_res_is_fresh():
    """本件是否已经出结果（用于区分「进行中」与「已出结果」）。"""
    with state_lock:
        return stats.get("result_for_item") == stats.get("items")


def fetch_active_product():
    """从 Java 拉取当前启用的工程配方编码；失败或未启用则返回 None。"""
    try:
        url = "%s/api/v1/recipe/active" % CFG.java_url.rstrip("/")
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        code = data.get("code")
        return code if code else None
    except Exception:
        return None


def current_product():
    """
    当前检测使用的配方编码。
    默认跟随「工程配方」页启用的配方；用 --product 显式指定时以命令行为准。
    """
    global ACTIVE_PRODUCT
    if CFG.product:                 # 命令行显式指定 → 固定使用
        return CFG.product
    code = fetch_active_product()
    if code and code != ACTIVE_PRODUCT:
        print("已切换到启用配方: %s" % code)
        ACTIVE_PRODUCT = code
    return ACTIVE_PRODUCT


def post_frame_to_java(jpg_bytes, frame_count):
    """把最清晰帧以 multipart 上传到 Java 检测接口。"""
    product = current_product()
    if not product:
        print("未指定配方：请在网页「工程配方」页启用一个配方，或用 --product 指定")
        return
    url = "%s/api/v1/inspect/%s/plate/frame" % (CFG.java_url.rstrip("/"), quote(product, safe=""))
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        passed = data.get("passed")
        set_tricolor(passed)   # OK→绿灯，NG→红灯（灯同步线程随后维持该状态）
        with state_lock:
            stats["sent"] += 1
            stats["result_for_item"] = stats["items"]   # 标记本件已出结果
            stats["last"] = {"passed": passed, "message": data.get("message"),
                             "screwMissing": data.get("screwMissing"),
                             "screwExpected": data.get("screwExpected"),
                             "frames": frame_count}
        print("[件#%d] 已检测 -> %s  %s (该件采 %d 帧)" %
              (stats["items"], "OK" if passed else "NG", data.get("message"), frame_count))
    except Exception as e:
        print("上传 Java 检测失败: %s" % e)


def _post_and_release(jpg_bytes, frame_count):
    """发送后释放 post_busy 锁，保证串行。"""
    try:
        post_frame_to_java(jpg_bytes, frame_count)
    finally:
        post_busy.release()


def _send_for_detect(best_gray, frames):
    """
    把一帧交给后台线程送去 Java 检测，返回是否真的发出去了（不打日志，由调用方决定）。

    <b>同一时刻只允许一个在途请求</b>（post_busy 信号量）：Java 单件检测约 300ms，
    不限流会越堆越多、最后全部超时。
    注意<b>先抢信号量再编码</b>：抢不到就直接返回，省掉一次白做的 JPEG 编码
    —— stream 模式下这个判断每秒会走好几次。
    """
    if not post_busy.acquire(blocking=False):
        return False
    try:
        okj, encj = cv2.imencode(".jpg", best_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not okj:
            post_busy.release()
            return False
        threading.Thread(target=_post_and_release,
                         args=(encj.tobytes(), frames), daemon=True).start()
        return True
    except Exception:
        post_busy.release()
        raise


def capture_loop():
    """
    连续取帧 + 分件状态机。

    这里有<b>两个相互独立</b>的判据，别混在一起：

    <b>--gate 决定"什么时候采图"</b>（本件正在进行时，哪些帧算数）：
      vision  只看画面：工件<b>完整进入视野</b>才开始采集，开始<b>离开视野</b>就结束本件（默认）
      sensor  只看光电传感器 LineStatus（原有行为）
      both    两者同时满足才采集（传感器定位置、视觉保证完整）

    <b>--rearm 决定"什么时候算下一件"</b>（上一件结束后如何解锁）：
      sensor  传感器先无料、再有料 = 下一件（默认，符合产线语义）
      vision  工件完全离开画面 = 下一件

    为什么要拆开：工件停在视野里不动时，画面永远不会"清空"，若用视野清场分件就会
    第一件之后再也不检测；而传感器的下降沿+上升沿才是真正的"换了一件"。
    """
    global latest_jpeg, fov_jpeg, sensor_active, fov_now, fov_box_now
    present = False
    best_gray = None
    best_sharp = -1.0
    frame_count = 0
    active_streak = 0      # 连续"在检测区"计数（进入防抖）
    inactive_streak = 0    # 连续"不在检测区"计数（离开防抖）
    item_start = 0.0       # 本件开始采集的时间戳（用于最长采集时长切件）
    need_clear = False     # 本件因"工件离开"结束后，要求先"清场"再接下一件（清场条件见 --rearm）
    last_state = None      # 上一帧的视野状态，用于只在变化时打日志
    sensor_unreadable_warned = False   # 传感器读不到时只警告一次，避免刷屏
    seg_frames = 0         # 距上次送检又攒了多少可用帧（stream 模式分段选最清晰帧）
    sent_in_item = 0       # 本件已送检次数
    last_detect = 0.0      # 本件内上次送检的时间戳
    last_preview = 0.0     # 上次编码 /grab 预览图的时间戳
    last_overlay = 0.0     # 上次编码 /fov 调试图的时间戳

    while True:
        st_frame = MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(st_frame), 0, ctypes.sizeof(st_frame))
        ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
        if ret != MV_OK:
            time.sleep(0.005)
            continue
        try:
            gray = frame_to_gray(st_frame)
            now = time.time()
            # /grab 预览图：<b>限流编码</b>。JPEG 编码一张 1200x1200 有实打实的开销，
            # 而预览只是给人看的，每秒几张足够。原先每帧都编码，白占掉本可以用来
            # 多采几帧工件的 CPU（工件完整露出的窗口很短，帧率就是有效样本数）。
            if now - last_preview >= CFG.preview_interval:
                ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    with state_lock:
                        latest_jpeg = enc.tobytes()
                last_preview = now

            sensor_raw = read_line_status()   # None = 读不到（未接/不支持），下面按"一直有料"退化
            sensor = sensor_raw
            if sensor is None:
                sensor = True  # 读不到传感器时，退化为“一直有料”（可用 /grab 手动）

            # 视野状态：sensor 模式下不需要，省掉这份计算
            if CFG.gate == "sensor":
                state, box = FOV_FULL, None
            else:
                state, box = fov_state(gray)
                if state != last_state:
                    if box is not None:
                        print("  视野状态 %s -> %s (工件框 %dx%d @(%d,%d), 占画面 %.1f%%)"
                              % (last_state, state, box[2], box[3], box[0], box[1],
                                 box[4] * 100))
                    else:
                        print("  视野状态 %s -> %s (未检出工件)" % (last_state, state))
                    last_state = state
                # /fov 调试图同样限流：它要画框再编码一次 JPEG，每帧都做很吃 CPU
                if CFG.fov_debug and now - last_overlay >= CFG.preview_interval:
                    dbg = fov_overlay_jpeg(gray, state, box)
                    if dbg:
                        with state_lock:
                            fov_jpeg = dbg
                    last_overlay = now

            with state_lock:
                sensor_active = sensor
                fov_now = state
                fov_box_now = box

            # ==== 两个<b>互不干扰</b>的信号 ====
            # item_on  = "一件正在进行"     —— 决定何时开始/结束一件（分件）
            # frame_ok = "这一帧可用于判定" —— 决定本件进行中哪些帧参与选最清晰帧（采图）
            #
            # 必须分开：视野状态会因为工件在边界上抖动、Otsu 把桌面/杂物当亮区而频繁跳变
            # （实测同一块板子逐帧框出 258x525、678x161 这种离谱结果）。
            # 若让视野去管分件，抖一下就结束本件；而它作为"帧过滤"是完全够用的
            # —— 视野不完整的帧图像本身就不全，不该拿去判定。
            if CFG.gate == "vision":
                item_on = (state == FOV_FULL)      # 兼容旧行为：纯视觉分件
            else:                                  # sensor / both
                item_on = sensor                   # 传感器管开始与结束
            if CFG.gate == "sensor":
                frame_ok = True                    # 不做视野过滤，整个有料期间的帧都算
            else:                                  # vision / both
                frame_ok = (state == FOV_FULL)

            # 传感器读不到却指望它分件 → 会永久没有工件，明确告警并退回视觉分件
            if CFG.gate != "vision" and sensor_raw is None and not sensor_unreadable_warned:
                print("警告: --gate %s 需要传感器分件，但读不到 LineStatus，"
                      "已退化为「一直有料」。请检查 --line/--active-high/接线；"
                      "台面调试可加 --max-seconds 3 按节拍切件。" % CFG.gate)
                sensor_unreadable_warned = True

            # 仅 --gate vision 才需要"清场"：视觉分件时工件后半截在画面里晃动会重复计件。
            # 传感器分件不需要 —— 下降沿本身就是无歧义的分界，结果一出来就能接下一件。
            if CFG.gate == "vision":
                if need_clear:
                    if CFG.rearm == "sensor" and sensor_raw is not None:
                        if not sensor_raw:
                            need_clear = False
                    elif state == FOV_NONE:
                        need_clear = False
                if need_clear:
                    item_on = False
            else:
                need_clear = False

            # 连续计数做防抖（只对"分件"信号做，帧过滤不需要防抖）
            if item_on:
                active_streak += 1
                inactive_streak = 0
            else:
                inactive_streak += 1
                active_streak = 0

            if not present:
                # 只有连续确认 enter_confirm 帧，才认定“新工件完整进入”，滤掉边界抖动
                if active_streak >= CFG.enter_confirm:
                    present = True
                    best_gray = None
                    best_sharp = -1.0
                    frame_count = 0
                    seg_frames = 0
                    sent_in_item = 0
                    last_detect = 0.0
                    item_start = time.time()
                    with state_lock:
                        stats["items"] += 1
                    set_tricolor(None)  # 工件进入→黄灯(运行中)，结果出来再变绿/红
                    print("[件#%d] %s，开始采集…" % (
                        stats["items"],
                        "工件已完整进入视野" if CFG.gate == "vision" else "传感器有料"))
            else:
                # 本件进行中：只把"视野完整"的帧拿去比清晰度。
                # 视野中途抖到 partial 只是这几帧不算，<b>不结束本件</b>。
                if frame_ok:
                    frame_count += 1
                    seg_frames += 1
                    s = sharpness(gray)
                    if s > best_sharp:
                        best_sharp = s
                        best_gray = gray.copy()

                # --detect-mode stream: 工件完整露出的窗口往往很短，不等本件结束就边采边送检。
                # 每 --detect-interval 毫秒送一次"这一小段里最清晰的一帧"，本件最多送
                # --detect-max 次。送完立刻重挑下一段的最清晰帧，避免整件都盯着同一张。
                if (CFG.detect_mode == "stream" and frame_ok and best_gray is not None
                        and seg_frames >= CFG.min_frames
                        and (CFG.detect_max <= 0 or sent_in_item < CFG.detect_max)
                        and (now - last_detect) * 1000.0 >= CFG.detect_interval):
                    if _send_for_detect(best_gray, seg_frames):
                        sent_in_item += 1
                        best_gray = None      # 下一段重新挑最清晰帧
                        best_sharp = -1.0
                        seg_frames = 0
                    # 成功与否都推进计时：Java 在忙时按同样间隔重试即可。
                    # 否则会每帧都试一次 —— 25fps 下就是每秒 25 次无谓尝试。
                    last_detect = now

                # 结束本件的两种情形：
                #  a) 分件信号消失（传感器无料 / 纯视觉模式下工件离开视野）
                #  b) 采集时长达到上限（工件长时间静止不动时按固定节拍切件，便于台面调试）
                leave = inactive_streak >= CFG.leave_confirm
                timeout = CFG.max_seconds > 0 and (time.time() - item_start) >= CFG.max_seconds
                if leave or timeout:
                    if leave:
                        reason = ("工件离开视野" if CFG.gate == "vision" else "传感器无料")
                    else:
                        reason = "采集达 %.0f 秒" % CFG.max_seconds
                    print("[件#%d] %s，结束本件(可用%d帧，已送检%d次)"
                          % (stats["items"], reason, frame_count, sent_in_item))
                    # 收尾再送一次：把最后一段（上次送检之后攒的帧）也检一遍。
                    # stream 模式下若这一段帧数不够就不送，避免拿零星几帧凑一次检测。
                    if best_gray is not None and seg_frames >= CFG.min_frames:
                        if not _send_for_detect(best_gray, seg_frames):
                            print("[件#%d] Java 检测繁忙，收尾这次送检跳过"
                                  % stats["items"])
                    elif sent_in_item == 0:
                        if frame_count > 0:
                            print("[件#%d] 可用帧不足(%d<%d)，丢弃。视野一直不完整？"
                                  % (stats["items"], frame_count, CFG.min_frames))
                        else:
                            print("[件#%d] 整件期间视野都不完整，无可用帧，丢弃"
                                  % stats["items"])

                    if timeout and item_on:
                        # 分件信号还在（台面静止调试）：立刻作为“新一件”继续，实现连续循环检测
                        best_gray = None
                        best_sharp = -1.0
                        frame_count = 0
                        seg_frames = 0
                        sent_in_item = 0
                        last_detect = 0.0
                        item_start = time.time()
                        with state_lock:
                            stats["items"] += 1
                        print("[件#%d] 分件信号仍在，开始新一件采集…" % stats["items"])
                    else:
                        present = False
                        # 只有纯视觉分件才需要"清场"：工件后半截在画面里晃动会重复计件。
                        # 传感器分件不需要 —— 无料就是分界，下次有料立刻就是新一件，
                        # 结果一出来就能接下一件，不用等画面清空。
                        if leave and CFG.gate == "vision":
                            need_clear = True
        except Exception as e:
            # 单帧异常不许搞垮整个采图线程（7×24 稳定性）
            print("采图处理异常(已忽略本帧): %s" % e)
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
                box = fov_box_now
                snap = {"sensorActive": sensor_active, "items": stats["items"],
                        "sent": stats["sent"], "last": stats["last"],
                        "gate": CFG.gate, "fovState": fov_now,
                        "fovBox": (None if box is None else
                                   {"x": box[0], "y": box[1], "w": box[2], "h": box[3],
                                    "areaRatio": round(box[4], 4)})}
            return self._json(200, snap)
        if path == "/fov":
            # 带视野状态标注的调试图（需 --fov-debug 1）。/grab 保持干净原图不受影响。
            with state_lock:
                jpg = fov_jpeg
            if jpg is None:
                return self._text(503, "尚无标注图（需启动参数 --fov-debug 1，且 --gate 非 sensor）")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
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
    ap.add_argument("--product", type=str, default=None,
                    help="固定使用的配方编码；不填则自动跟随网页「工程配方」页启用的配方")
    ap.add_argument("--java-url", type=str, default="http://127.0.0.1:8088")
    ap.add_argument("--line", type=str, default="Line0", help="传感器输入线，如 Line0/Line1")
    ap.add_argument("--active-high", type=int, default=1, help="1=高电平有料，0=低电平有料")
    ap.add_argument("--gate", type=str, default="both", choices=["vision", "sensor", "both"],
                    help="分件与采图的判据组合（推荐 both）："
                         "both=传感器管一件的开始/结束，视野只用来筛掉不完整的帧(默认)；"
                         "sensor=传感器管开始/结束，不做视野筛帧；"
                         "vision=纯视觉分件(工件完整进入才开始、离开就结束)，"
                         "台面上工件不走时需配合 --rearm/--max-seconds")
    ap.add_argument("--fov-margin", type=int, default=12,
                    help="视野余量(像素)：工件外接框四边都离画面边界至少这么远，才算「完整进入」。"
                         "太小会在齐边时反复跳状态，太大则工件要离边界很远才肯采图")
    ap.add_argument("--fov-min-area", type=float, default=0.05,
                    help="工件最小面积占画面比例，低于此视为画面里没有工件(滤掉空传送带/噪点)")
    ap.add_argument("--fov-debug", type=int, default=0,
                    help="1=生成带视野状态标注的调试图，浏览器开 http://127.0.0.1:<port>/fov 查看")
    ap.add_argument("--strobe", type=int, default=1, help="1=启用补光频闪输出(曝光时打光)，0=不控制光源")
    ap.add_argument("--strobe-line", type=str, default="Line1", help="补光灯所接的相机输出线，如 Line1/Line2")
    ap.add_argument("--strobe-duration", type=int, default=0, help="频闪脉宽(微秒)，0=跟随曝光时长")
    ap.add_argument("--strobe-delay", type=int, default=0, help="频闪延迟(微秒)")
    ap.add_argument("--rearm", type=str, default="sensor", choices=["sensor", "vision"],
                    help="一件结束后靠什么解锁下一件（与 --gate 相互独立：--gate 决定何时采图，"
                         "--rearm 决定何时算新的一件）。"
                         "sensor=传感器先无料、再有料就是下一件(推荐，工件停在视野里也能分件)；"
                         "vision=工件完全离开画面才算下一件(台面调试时工件不走，会卡住)")
    ap.add_argument("--detect-mode", type=str, default="end", choices=["end", "stream"],
                    help="何时送检：end=本件结束时只送最清晰的一帧(默认)；"
                         "stream=本件进行中边采边送，工件完整露出的窗口很短时用这个，"
                         "能在窗口内拿到多次检测结果")
    ap.add_argument("--detect-interval", type=float, default=250,
                    help="stream 模式下两次送检的最小间隔(毫秒)。Java 单件约 300ms，"
                         "设太小只会被在途限流挡掉，没有意义")
    ap.add_argument("--detect-max", type=int, default=3,
                    help="stream 模式下单件最多送检几次，0=不限。注意每次送检在网页统计里"
                         "都算一次检测，会放大产量计数")
    ap.add_argument("--preview-interval", type=float, default=0.2,
                    help="/grab 与 /fov 预览图的编码间隔(秒)。预览只是给人看的，"
                         "调大可把 CPU 留给多采几帧工件")
    ap.add_argument("--min-frames", type=int, default=2,
                    help="一件至少要有多少「视野完整」的帧才判定，少于此数丢弃。"
                         "工件走得快、完整露出的窗口只有零点几秒时，可用帧本来就不多，"
                         "设 3 会把大量本可判定的工件丢掉；2 帧已够挑出较清晰的一张")
    ap.add_argument("--enter-confirm", type=int, default=3,
                    help="连续多少帧满足条件才确认工件进入(防抖)。工件速度快时调小，否则会错过完整进入的窗口")
    ap.add_argument("--leave-confirm", type=int, default=4,
                    help="连续多少帧不满足条件才确认工件离开(防抖)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="单件最长采集秒数：工件长时间停在视野内时，每满该秒数自动结束本件并开始新一件；"
                         "0=不限(仅靠进入/离开分件)。流水线上工件会自己走过，一般设 0")
    ap.add_argument("--max-width", type=int, default=1200,
                    help="输出图像最大宽度：相机原始5120过大会让螺丝候选点暴增、配准不稳，"
                         "在源头缩到此宽度(默认1200，螺丝约7px)。0=不缩放")
    ap.add_argument("--light", type=int, default=1, help="1=启用三色灯(继电器)，0=不控制")
    ap.add_argument("--relay-port", type=str, default="COM7", help="继电器模块串口，如 COM7")
    ap.add_argument("--relay-baud", type=int, default=9600, help="继电器波特率(常见9600/115200)")
    ap.add_argument("--beep", type=int, default=1, help="NG 时是否鸣蜂鸣器(out4)，1=响 0=不响")
    CFG = ap.parse_args()
    CFG.active_high = bool(CFG.active_high)
    CFG.fov_debug = bool(CFG.fov_debug)
    if CFG.gate == "vision":
        print("分件依据: 视野（工件完整进入=开始，离开=结束）")
        print("解锁下一件: --rearm %s（%s）" % (
            CFG.rearm,
            "传感器先无料、再有料" if CFG.rearm == "sensor" else "工件完全离开画面"))
    else:
        print("分件依据: 传感器 %s（有料=开始一件，无料=结束本件；结果一出即可接下一件）"
              % CFG.line)
    if CFG.detect_mode == "stream":
        print("送检方式: stream（本件进行中每 %.0fms 送一次，最多 %s 次；"
              "注意网页产量按检测次数计）"
              % (CFG.detect_interval, "不限" if CFG.detect_max <= 0 else CFG.detect_max))
    else:
        print("送检方式: end（本件结束时送最清晰的一帧）")
    if CFG.gate == "sensor":
        print("采图筛帧: 不筛（有料期间所有帧都参与选最清晰帧）")
    else:
        print("采图筛帧: 视野完整的帧才参与（外接框四边距边界 ≥%dpx 且面积占比 ≥%.1f%%）；"
              "视野中途抖动只是丢几帧，不会结束本件"
              % (CFG.fov_margin, CFG.fov_min_area * 100))

    open_camera(CFG)
    relay_open()       # 打开继电器串口（三色灯）
    light_selftest()   # 开机自检：黄→绿→红各亮1秒，确认灯控制是否生效
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=light_sync_loop, daemon=True).start()  # 三色灯跟随界面状态
    try:
        server = ThreadingHTTPServer(("127.0.0.1", CFG.port), Handler)
        print("采图服务已启动: http://127.0.0.1:%d  (/health /grab /status /fov)" % CFG.port)
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
