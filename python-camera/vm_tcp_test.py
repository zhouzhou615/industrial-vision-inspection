# -*- coding: utf-8 -*-
"""
模拟海康 VisionMaster 通过 TCP 向 Java 推送检测结果，用于联调接收端 + 网页看板。

报文格式（分号分隔，UTF-8，\n 结尾，与 VmResultServer 约定一致）：
    产品编码;结果;螺丝总数;漏打数;Logo结果;Logo角度;标注图路径;备注

用法：
    python vm_tcp_test.py                       # 默认发 5 件到 127.0.0.1:9000
    python vm_tcp_test.py --host 127.0.0.1 --port 9000 --product SKU-001 --count 10 --interval 1.5
    python vm_tcp_test.py --once "SKU-001;NG;9;1;OK;0.3;;S5漏打"   # 只发一行自定义报文
"""
import argparse
import random
import socket
import time


def send_lines(host, port, lines, interval):
    with socket.create_connection((host, port), timeout=5) as s:
        print("已连接 %s:%d" % (host, port))
        for ln in lines:
            data = (ln + "\n").encode("utf-8")
            s.sendall(data)
            print("已发送: %s" % ln)
            time.sleep(interval)
    print("发送完毕。")


def build_random(product, i):
    """随机生成一件结果：约 40% 概率 NG。"""
    total = 9
    if random.random() < 0.4:
        missing = random.randint(1, 3)
        logo = random.choice(["OK", "NG"])
        angle = round(random.uniform(-6, 6), 1)
        detail = "S%d漏打" % random.randint(1, total) if missing else "Logo异常"
        return "%s;NG;%d;%d;%s;%s;;%s" % (product, total, missing, logo, angle, detail)
    else:
        return "%s;OK;%d;0;OK;%s;;" % (product, total, round(random.uniform(-1, 1), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--product", default="SKU-001")
    ap.add_argument("--count", type=int, default=5, help="模拟发送的工件数")
    ap.add_argument("--interval", type=float, default=1.5, help="每件间隔秒")
    ap.add_argument("--once", default=None, help="只发送这一行自定义报文")
    args = ap.parse_args()

    if args.once:
        send_lines(args.host, args.port, [args.once], 0)
        return

    lines = [build_random(args.product, i) for i in range(args.count)]
    send_lines(args.host, args.port, lines, args.interval)


if __name__ == "__main__":
    main()
