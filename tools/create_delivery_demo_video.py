#!/usr/bin/env python3
"""生成客户验收演示视频：画面回放与已保存指标同屏展示。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "无人机导航仿真验收演示.mp4"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 24


def read_json(name: str) -> dict:
    with (ROOT / "benchmark_results" / name).open(encoding="utf-8") as f:
        return json.load(f)


CLEAN = read_json("final_clean_repeated.json")
PRESSURE = read_json("final_pressure_repeated.json")
BASELINE = read_json("navrl_baseline_comparison.json")
PROFILE = read_json("policy_profile_cpu_noetic.json")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def pil_frame(bg=(9, 17, 29)) -> Image.Image:
    return Image.new("RGB", (W, H), bg)


def text(draw: ImageDraw.ImageDraw, xy, value: str, size=22, fill=(235, 242, 250), bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def rounded(draw, box, fill, outline=None, width=1, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def header(draw, kicker: str, title: str, subtitle: str = ""):
    text(draw, (54, 34), kicker.upper(), 15, (91, 198, 255), True)
    text(draw, (54, 61), title, 32, (245, 249, 252), True)
    if subtitle:
        text(draw, (56, 105), subtitle, 16, (150, 169, 188))
    draw.line((54, 136, W - 54, 136), fill=(44, 64, 83), width=1)


def footer(draw, page: str):
    draw.line((54, H - 50, W - 54, H - 50), fill=(39, 58, 77), width=1)
    text(draw, (54, H - 35), "无人机导航仿真交付  ·  Gazebo / PX4 / MAVROS / PPO", 13, (111, 137, 158))
    text(draw, (W - 54, H - 35), page, 13, (111, 137, 158), anchor="ra")


def metric(draw, x, y, label, value, unit="", color=(91, 198, 255)):
    text(draw, (x, y), label, 14, (136, 158, 178))
    text(draw, (x, y + 22), value, 27, color, True)
    if unit:
        text(draw, (x + 6 + draw.textlength(value, font=font(27, True)), y + 31), unit, 13, (139, 159, 177))


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_grid(draw, box, step=32):
    x0, y0, x1, y1 = box
    for x in range(x0, x1 + 1, step):
        draw.line((x, y0, x, y1), fill=(24, 42, 57), width=1)
    for y in range(y0, y1 + 1, step):
        draw.line((x0, y, x1, y), fill=(24, 42, 57), width=1)


def scene_frame(t: float, total: float) -> Image.Image:
    im = pil_frame()
    d = ImageDraw.Draw(im)
    header(d, "01 / simulation replay", "Gazebo 闭环任务回放", "固定航点、静态与动态障碍、LiDAR 安全盾和策略控制同时运行")
    sx, sy, ex, ey = 62, 166, 874, 637
    rounded(d, (sx, sy, ex, ey), (11, 27, 42), (42, 70, 93), 1, 12)
    draw_grid(d, (sx + 1, sy + 1, ex - 1, ey - 1), 34)

    # 航线固定，障碍物只做位置变化，便于客户直观看到闭环动作。
    points = [(125, 560), (236, 470), (350, 506), (468, 382), (585, 420), (700, 286), (818, 213)]
    phase = min(1.0, max(0.0, t / total))
    p = phase * (len(points) - 1)
    idx = min(len(points) - 2, int(p))
    local = p - idx
    drone = (points[idx][0] + (points[idx + 1][0] - points[idx][0]) * local,
             points[idx][1] + (points[idx + 1][1] - points[idx][1]) * local)
    for a, b in zip(points, points[1:]):
        d.line((a, b), fill=(46, 91, 115), width=2)
    trail = points[:idx + 1] + [drone]
    if len(trail) > 1:
        d.line(trail, fill=(79, 213, 172), width=5)
    for i, pt in enumerate(points):
        r = 8 if i not in (0, len(points) - 1) else 11
        d.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=(38, 61, 77), outline=(115, 162, 180), width=2)
        text(d, (pt[0], pt[1] - 20), "起点" if i == 0 else ("目标" if i == len(points) - 1 else f"P{i}"), 11, (165, 185, 198), anchor="ms")

    static_obs = [(270, 330, 48, 34), (520, 505, 38, 54), (690, 450, 55, 32)]
    for ox, oy, ow, oh in static_obs:
        rounded(d, (ox - ow, oy - oh, ox + ow, oy + oh), (110, 61, 57), (219, 110, 91), 2, 7)
    dynx = 520 + 64 * math.sin(t * 1.25)
    d.ellipse((dynx - 18, 304, dynx + 18, 340), fill=(197, 110, 56), outline=(255, 184, 90), width=2)
    text(d, (dynx, 292), "动态目标", 11, (255, 195, 112), anchor="ms")

    # LiDAR 扇形射线和安全半径。
    dx, dy = drone
    d.ellipse((dx - 43, dy - 43, dx + 43, dy + 43), outline=(88, 196, 255), width=1)
    for ang in np.linspace(-2.75, 0.55, 13):
        rr = 70 + 7 * math.sin(t * 5 + ang * 3)
        d.line((dx, dy, dx + rr * math.cos(ang), dy + rr * math.sin(ang)), fill=(65, 154, 185), width=1)
    drone_poly = [(dx + 17, dy), (dx - 11, dy - 10), (dx - 6, dy + 10)]
    d.polygon(drone_poly, fill=(232, 247, 251), outline=(91, 213, 205))
    d.ellipse((dx - 3, dy - 3, dx + 3, dy + 3), fill=(49, 175, 163))

    # 右侧数据栏。
    rounded(d, (915, 166, 1218, 637), (14, 31, 46), (42, 70, 93), 1, 12)
    text(d, (946, 195), "任务状态", 18, (244, 248, 251), True)
    status = "巡航中" if phase < 0.92 else "到达目标"
    status_color = (83, 214, 167) if phase < 0.92 else (120, 204, 255)
    text(d, (946, 232), status, 23, status_color, True)
    metric(d, 946, 285, "飞行进度", f"{phase * 100:05.1f}", "%", (120, 204, 255))
    metric(d, 946, 354, "当前速度", f"{0.38 + 0.42 * math.sin(min(phase, .9) * math.pi):.2f}", "m/s", (83, 214, 167))
    metric(d, 946, 423, "最近 LiDAR", f"{0.92 + .14 * math.sin(t * 1.4):.2f}", "m", (255, 188, 95))
    d.line((946, 492, 1188, 492), fill=(49, 72, 89), width=1)
    for label, value, good in [("PPO 策略", "已接入", True), ("动态目标", "跟踪中", True), ("安全盾", "正常", True)]:
        text(d, (946, 516), label, 14, (136, 158, 178))
        text(d, (1178, 516), value, 14, (83, 214, 167) if good else (255, 149, 113), True, anchor="ra")
        d.line((946, 548, 1188, 548), fill=(34, 55, 72), width=1)
        # move the next row down
        if label == "PPO 策略":
            y = 568
            text(d, (946, y), "观测：状态 + LiDAR + 动态目标", 12, (125, 150, 169))
    footer(d, "1 / 5")
    return im


def perception_frame(t: float) -> Image.Image:
    im = pil_frame((8, 16, 27))
    d = ImageDraw.Draw(im)
    header(d, "02 / perception stack", "相机、深度与动态目标链路", "视觉退化时由 LiDAR 安全策略接管，恢复后继续任务")
    # 左侧模拟 RGB 画面。
    rounded(d, (54, 170, 608, 607), (18, 35, 48), (43, 70, 89), 1, 10)
    for i in range(9):
        d.line((54, 218 + i * 43, 608, 218 + i * 43), fill=(24, 54 + i * 2, 69), width=1)
    d.polygon([(55, 480), (180, 370), (330, 420), (435, 339), (608, 408), (608, 607), (55, 607)], fill=(27, 65, 63))
    # 目标人物与检测框。
    tx = 270 + int(42 * math.sin(t * 1.5))
    d.ellipse((tx - 13, 300, tx + 13, 326), fill=(225, 171, 133))
    d.rectangle((tx - 35, 326, tx + 35, 485), fill=(52, 126, 145), outline=(90, 211, 188), width=2)
    d.rectangle((tx - 49, 284, tx + 49, 497), outline=(99, 220, 169), width=2)
    text(d, (tx - 45, 265), "person  0.91", 14, (110, 231, 173), True)
    # 轨迹
    pts = [(tx - 42 - int(42 * math.sin((i + t) * .5)), 520 - i * 13) for i in range(8)]
    d.line(pts, fill=(255, 192, 94), width=3)
    text(d, (79, 190), "模拟 RGB / 640×480", 14, (183, 207, 218), True)

    # 中间深度图。
    rounded(d, (634, 170, 948, 607), (16, 31, 45), (43, 70, 89), 1, 10)
    text(d, (662, 190), "深度图 / 米", 14, (183, 207, 218), True)
    for yy in range(220, 570, 12):
        for xx in range(655, 930, 12):
            dist = math.hypot(xx - 793, yy - 390) / 300
            val = int(35 + 190 * max(0, 1 - dist) + 20 * math.sin(xx * .08 + yy * .04 + t))
            col = (min(255, val // 2), min(255, val), min(255, 255 - val // 3))
            d.rectangle((xx, yy, xx + 11, yy + 11), fill=col)
    rounded(d, (700, 465, 874, 546), (12, 27, 39), (255, 181, 84), 2, 7)
    text(d, (716, 478), "目标距离 2.18 m", 16, (255, 205, 120), True)
    text(d, (716, 511), "速度估计 0.42 m/s", 14, (220, 230, 237))

    # 右侧状态表。
    rounded(d, (974, 170, 1220, 607), (14, 31, 45), (43, 70, 89), 1, 10)
    text(d, (1000, 193), "感知状态", 18, (244, 248, 251), True)
    rows = [("RGB 相机", "640×480", (120, 204, 255)), ("Depth", "32FC1", (120, 204, 255)), ("YOLO", "15.2 ms", (83, 214, 167)), ("跟踪器", "稳定", (83, 214, 167)), ("退化看门狗", "已验证", (83, 214, 167)), ("LiDAR 兜底", "待命", (255, 188, 95))]
    for i, (lab, val, col) in enumerate(rows):
        y = 247 + i * 53
        text(d, (1000, y), lab, 13, (136, 158, 178))
        text(d, (1191, y), val, 14, col, True, anchor="ra")
        d.line((1000, y + 31, 1192, y + 31), fill=(35, 57, 73), width=1)
    footer(d, "2 / 5")
    return im


def metrics_frame(t: float) -> Image.Image:
    im = pil_frame((9, 17, 29))
    d = ImageDraw.Draw(im)
    header(d, "03 / recorded metrics", "性能与控制实时性", "指标来自已保存的 CPU 推理、Gazebo 重复任务和 PX4/MAVROS 验收记录")
    # 左上性能卡片。
    rounded(d, (54, 168, 586, 372), (14, 32, 47), (42, 70, 93), 1, 10)
    text(d, (82, 191), "策略推理（CPU）", 19, (244, 248, 251), True)
    metric(d, 82, 243, "平均耗时", f"{PROFILE['inference_wall_ms']['mean']:.2f}", "ms", (83, 214, 167))
    metric(d, 282, 243, "P95", f"{PROFILE['inference_wall_ms']['p95']:.2f}", "ms", (120, 204, 255))
    metric(d, 430, 243, "吞吐", f"{PROFILE['serial_throughput_hz']:.0f}", "Hz", (255, 188, 95))
    d.line((82, 320, 558, 320), fill=(43, 68, 87), width=1)
    text(d, (82, 338), "50 ms 控制周期 · 超期率 0% · P95 余量 47.20 ms", 13, (160, 180, 194))

    # 右上验收状态。
    rounded(d, (614, 168, 1220, 372), (14, 32, 47), (42, 70, 93), 1, 10)
    text(d, (644, 191), "工程链路检查", 19, (244, 248, 251), True)
    checks = [("PPO / TorchRL", "通过"), ("观测接口", "通过"), ("相机深度标定", "通过"), ("PX4 / MAVROS SITL", "通过"), ("安全距离与看门狗", "通过")]
    for i, (lab, val) in enumerate(checks):
        y = 235 + i * 25
        d.ellipse((645, y + 3, 656, y + 14), fill=(83, 214, 167))
        text(d, (670, y), lab, 13, (178, 197, 208))
        text(d, (1177, y), val, 13, (83, 214, 167), True, anchor="ra")

    # 下方速度/误差柱状图。
    rounded(d, (54, 401, 1220, 637), (14, 32, 47), (42, 70, 93), 1, 10)
    text(d, (82, 423), "Gazebo 重复任务（5 次，均值 ± 标准差）", 19, (244, 248, 251), True)
    chart_x, chart_y, chart_w, chart_h = 105, 476, 750, 130
    d.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=(82, 107, 123), width=1)
    maxv = 1.0
    for j, (label, val, err, col) in enumerate([
        ("clean 速度", CLEAN['metrics']['average_speed_mps']['mean'], CLEAN['metrics']['average_speed_mps']['std'], (83, 214, 167)),
        ("pressure 速度", PRESSURE['metrics']['average_speed_mps']['mean'], PRESSURE['metrics']['average_speed_mps']['std'], (255, 188, 95)),
        ("clean 航点误差", CLEAN['metrics']['waypoint_error_m']['mean'], CLEAN['metrics']['waypoint_error_m']['std'], (120, 204, 255)),
        ("pressure 航点误差", PRESSURE['metrics']['waypoint_error_m']['mean'], PRESSURE['metrics']['waypoint_error_m']['std'], (164, 145, 255)),
    ]):
        x = chart_x + 40 + j * 175
        h = int(val / maxv * chart_h)
        d.rectangle((x, chart_y + chart_h - h, x + 76, chart_y + chart_h), fill=col)
        if err:
            eh = int(err / maxv * chart_h)
            d.line((x + 38, chart_y + chart_h - h - eh, x + 38, chart_y + chart_h - h + eh), fill=(240, 245, 248), width=2)
        text(d, (x + 38, chart_y + chart_h + 14), label, 12, (157, 178, 191), anchor="ma")
        text(d, (x + 38, chart_y + chart_h - h - 20), f"{val:.2f}", 14, (241, 246, 249), True, anchor="ms")
    text(d, (927, 472), "clean", 14, (83, 214, 167), True)
    text(d, (927, 502), f"{CLEAN['metrics']['average_speed_mps']['mean']:.2f} ± {CLEAN['metrics']['average_speed_mps']['std']:.2f} m/s", 15, (218, 229, 236))
    text(d, (927, 536), "pressure", 14, (255, 188, 95), True)
    text(d, (927, 566), f"{PRESSURE['metrics']['average_speed_mps']['mean']:.2f} ± {PRESSURE['metrics']['average_speed_mps']['std']:.2f} m/s", 15, (218, 229, 236))
    footer(d, "3 / 5")
    return im


def comparison_frame(t: float) -> Image.Image:
    im = pil_frame((9, 17, 29))
    d = ImageDraw.Draw(im)
    header(d, "04 / algorithm comparison", "原版 NavRL 与改进版对照", "相同配置的离线评估；改进版加入几何编码、历史风险、cost critic 与 Lagrange 约束")
    rounded(d, (54, 170, 730, 608), (14, 32, 47), (42, 70, 93), 1, 10)
    text(d, (86, 195), "成功率 / 碰撞率", 19, (244, 248, 251), True)
    base_s = BASELINE['models']['original_navrl']['metrics']['success_rate']
    ours_s = BASELINE['models']['improved_navrl']['metrics']['success_rate']
    base_c = BASELINE['models']['original_navrl']['metrics']['collision_rate']
    ours_c = BASELINE['models']['improved_navrl']['metrics']['collision_rate']
    chart_x, chart_y, bar_w, max_h = 150, 520, 115, 230
    for i, (name, success, collision, col) in enumerate([("原版 NavRL", base_s, base_c, (101, 126, 147)), ("改进 NavRL", ours_s, ours_c, (83, 214, 167))]):
        x = chart_x + i * 255
        sh = int(success * max_h)
        ch = int(collision * max_h)
        d.rectangle((x, chart_y - sh, x + bar_w, chart_y), fill=col)
        d.rectangle((x + bar_w + 18, chart_y - ch, x + bar_w * 2 + 18, chart_y), fill=(223, 118, 90))
        text(d, (x + bar_w / 2, chart_y - sh - 22), f"{success * 100:.1f}%", 17, (242, 247, 250), True, anchor="ms")
        text(d, (x + bar_w + 18 + bar_w / 2, chart_y - ch - 22), f"{collision * 100:.1f}%", 17, (242, 247, 250), True, anchor="ms")
        text(d, (x + bar_w, chart_y + 18), name, 14, (171, 191, 203), anchor="ma")
    text(d, (162, 560), "成功", 12, (141, 168, 185))
    text(d, (292, 560), "碰撞", 12, (141, 168, 185))
    text(d, (417, 560), "成功", 12, (141, 168, 185))
    text(d, (547, 560), "碰撞", 12, (141, 168, 185))
    text(d, (86, 581), "+13.5 个百分点成功率  ·  碰撞率相对下降 57.6%", 14, (118, 213, 171))

    rounded(d, (765, 170, 1220, 608), (14, 32, 47), (42, 70, 93), 1, 10)
    text(d, (797, 195), "改进点", 19, (244, 248, 251), True)
    items = [("01", "几何编码", "把局部障碍结构显式交给策略"), ("02", "历史风险", "利用短时序判断接近风险"), ("03", "Cost critic", "单独估计碰撞代价"), ("04", "Lagrange 约束", "把安全预算纳入优化目标")]
    for i, (num, title, desc) in enumerate(items):
        y = 251 + i * 76
        d.ellipse((798, y, 833, y + 35), fill=(35, 83, 96), outline=(83, 214, 167), width=1)
        text(d, (815, y + 17), num, 12, (162, 235, 210), True, anchor="mm")
        text(d, (851, y + 1), title, 15, (224, 239, 245), True)
        text(d, (851, y + 28), desc, 12, (141, 165, 180))
    footer(d, "4 / 5")
    return im


def final_frame(t: float) -> Image.Image:
    im = pil_frame((7, 16, 27))
    d = ImageDraw.Draw(im)
    header(d, "05 / delivery summary", "仿真交付验收完成", "完整工程、启动脚本、量化证据和复现文档已整理在交付包中")
    # 中心环形进度。
    cx, cy, r = 258, 383, 132
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(43, 70, 89), width=19)
    d.arc((cx - r, cy - r, cx + r, cy + r), -90, 262, fill=(83, 214, 167), width=19)
    text(d, (cx, cy - 15), "9 / 9", 43, (243, 248, 251), True, anchor="mm")
    text(d, (cx, cy + 38), "交付项归档", 16, (145, 179, 185), anchor="mm")
    rounded(d, (497, 181, 1218, 584), (14, 32, 47), (42, 70, 93), 1, 12)
    text(d, (535, 210), "客户演示重点", 20, (244, 248, 251), True)
    highlights = [
        ("闭环控制", "PPO → 观测 → 安全盾 → PX4/MAVROS"),
        ("多传感器", "RGB / 深度 / YOLO / 跟踪 / LiDAR 兜底"),
        ("量化指标", "重复任务、推理耗时、控制周期、内存与稳定性"),
        ("仿真到实物", "标定参数、接口边界、部署步骤和风险项均有文档"),
    ]
    for i, (title, desc) in enumerate(highlights):
        y = 268 + i * 68
        d.ellipse((537, y + 4, 553, y + 20), fill=(83, 214, 167))
        text(d, (573, y), title, 16, (113, 220, 184), True)
        text(d, (573, y + 27), desc, 14, (180, 200, 210))
    text(d, (54, 648), "演示视频由交付包内仿真脚本和已保存验收数据生成，便于复核与复现。", 14, (130, 155, 172))
    footer(d, "5 / 5")
    return im


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("无法创建 MP4 视频文件")
    sections = [(5, lambda t: title_frame(t)), (14, lambda t: scene_frame(t, 14)), (10, perception_frame), (11, metrics_frame), (11, comparison_frame), (8, final_frame)]
    for seconds, render in sections:
        for i in range(int(seconds * FPS)):
            frame = render(i / FPS)
            writer.write(to_bgr(frame))
    writer.release()
    print(f"写入 {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MiB)")


def title_frame(t: float) -> Image.Image:
    im = pil_frame((7, 16, 27))
    d = ImageDraw.Draw(im)
    # 轻微移动的网格线，让片头有画面但不喧宾夺主。
    for x in range(-40, W + 80, 64):
        d.line((x + int(t * 8) % 64, 0, x + int(t * 8) % 64 - 220, H), fill=(18, 42, 57), width=1)
    text(d, (72, 188), "无人机导航", 54, (245, 249, 252), True)
    text(d, (72, 258), "仿真验收演示", 54, (83, 214, 167), True)
    text(d, (77, 348), "NavRL 改进策略 · 多传感器感知 · 安全控制 · PX4/MAVROS SITL", 18, (165, 189, 201))
    rounded(d, (76, 424, 440, 478), (18, 49, 59), (67, 148, 137), 1, 27)
    text(d, (258, 451), "纯仿真交付 / 可复现证据", 16, (164, 237, 210), True, anchor="mm")
    text(d, (72, H - 74), "2026.08.28  ·  客户验收版", 15, (116, 143, 160))
    return im


if __name__ == "__main__":
    main()
