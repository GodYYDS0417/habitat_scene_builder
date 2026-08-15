#!/usr/bin/env python3
"""make_villa_plan.py — 按真实户型工程数据生成两层别墅的放置方案 JSON。

布局参考安居客 94㎡ 2室2厅户型（客厅开间 5.15m、沙发 3x0.8、电视柜 3x0.5、
茶几 2x0.6、餐桌 2x0.8、床 2x2m、床头柜 0.5x0.5、衣柜 3x0.6）：
  F1 动区：客厅(沙发贴西墙-茶几-电视柜东墙一线) + 餐厅(餐桌+4椅) + 厨房(操作台)
           + 玄关(鞋柜) + 书房(书桌+书架) + 电梯(轿厢停一楼)
  F2 静区：主卧(床+双床头柜+衣柜) + 次卧(床+床头柜+衣柜) + 书房(书桌+书架)
           + 茶室(桌+2椅)

输出: villa_plan.json —— 直接用 habitat_scene_builder_v2.py --plan-in 重建；
      也可以手工改里面的 translation/rotation/motion_type。

用法:
    python make_villa_plan.py --out villa_plan.json [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

F2 = 2.9            # 二层地面高度（层高 2.9m）
TABLE_TOP = 0.75    # 餐桌/书桌台面
COFFEE_TOP = 0.42   # 茶几面
STAND_TOP = 0.45    # 电视柜面
COUNTER_TOP = 0.86  # 厨房台面
NIGHT_TOP = 0.50    # 床头柜面
SHELF_TOPS = (0.415, 0.815, 1.215)   # 书架搁板面


def quat_yaw(yaw: float) -> list[float]:
    return [round(math.cos(yaw / 2), 6), 0.0, round(math.sin(yaw / 2), 6), 0.0]


def put(cls: str, x: float, y: float, z: float, yaw: float = 0.0,
        motion: str = "STATIC", rule: str = "floorplan") -> dict:
    return {"class": cls,
            "translation": [round(x, 4), round(y, 4), round(z, 4)],
            "rotation": quat_yaw(yaw),
            "motion_type": motion,
            "rule": rule}


def face(cls: str, x: float, y: float, z: float,
         tx: float, tz: float, motion: str = "STATIC") -> dict:
    """摆放并朝向 (tx, tz)（椅子面向桌子等）。"""
    return put(cls, x, y, z, math.atan2(tx - x, tz - z), motion)


def build_plan(rng: random.Random) -> list[dict]:
    P: list[dict] = []

    # ================= F1 客厅（沙发西墙 / 电视东墙 / 茶几居中）=================
    # 电视墙 = 客厅/书房隔墙 x=1.2，门洞 z[3.9,4.8]（南端，真实户型门开墙端）。
    # 客厅组合居中 z=1.7：电视柜 z[0.5,2.9] 与门洞(侵蚀后 z≥4.2)净距 1.0m，
    # 沙发 z[0.2,3.2] 贴西墙，茶几在沙发正东 0.45m —— 电视正对沙发。
    P.append(put("tv_stand", 0.90, 0, 1.70, math.pi / 2))         # 电视柜贴东隔墙
    P.append(put("tv", 0.90, STAND_TOP, 1.70, -math.pi / 2))      # 电视面向沙发
    P.append(put("sofa", -4.55, 0, 1.70, math.pi / 2))            # 沙发贴西墙，面向电视
    P.append(put("coffee_table", -3.32, 0, 1.70, math.pi / 2))    # 茶几距沙发前沿 0.45m
    # 茶几上的小物件（可交互）
    P.append(put("bowl", -3.47, COFFEE_TOP, 1.55, 0, "DYNAMIC"))
    P.append(put("apple", -3.20, COFFEE_TOP, 1.60, 0, "DYNAMIC"))
    P.append(put("apple", -3.16, COFFEE_TOP, 1.80, 0, "DYNAMIC"))
    P.append(put("cup", -3.57, COFFEE_TOP, 1.86, 0, "DYNAMIC"))
    P.append(put("cup", -3.08, COFFEE_TOP, 1.46, 0, "DYNAMIC"))
    P.append(put("book", -3.32, COFFEE_TOP, 1.82, 0.4, "DYNAMIC"))
    P.append(put("lamp", -4.55, 0, 4.40, 0, "DYNAMIC"))           # 落地灯(沙发角)

    # ================= F1 餐厅（餐桌 + 4 椅）=================================
    tx, tz = -1.5, -1.1
    P.append(put("table", tx, 0, tz))
    P.append(face("chair", -1.9, 0, -0.325, tx, tz))
    P.append(face("chair", -1.1, 0, -0.325, tx, tz))
    P.append(face("chair", -1.9, 0, -1.875, tx, tz))
    P.append(face("chair", -1.1, 0, -1.875, tx, tz))
    P.append(put("plate", -1.9, TABLE_TOP, -0.92, 0, "DYNAMIC"))
    P.append(put("plate", -1.1, TABLE_TOP, -0.92, 0, "DYNAMIC"))
    P.append(put("plate", -1.9, TABLE_TOP, -1.28, 0, "DYNAMIC"))
    P.append(put("plate", -1.1, TABLE_TOP, -1.28, 0, "DYNAMIC"))
    P.append(put("cup", -1.62, TABLE_TOP, -0.86, 0, "DYNAMIC"))
    P.append(put("cup", -1.38, TABLE_TOP, -1.34, 0, "DYNAMIC"))
    P.append(put("bottle", -1.76, TABLE_TOP, -1.10, 0, "DYNAMIC"))
    P.append(put("bowl", -1.24, TABLE_TOP, -1.10, 0, "DYNAMIC"))

    # ================= F1 厨房（操作台贴北墙）===============================
    P.append(put("counter", -3.5, 0, -4.53))
    P.append(put("bottle", -4.20, COUNTER_TOP, -4.50, 0, "DYNAMIC"))
    P.append(put("bottle", -3.90, COUNTER_TOP, -4.56, 0, "DYNAMIC"))
    P.append(put("plate", -3.30, COUNTER_TOP, -4.50, 0, "DYNAMIC"))
    P.append(put("box", -2.85, COUNTER_TOP, -4.50, 0.3, "DYNAMIC"))

    # ================= F1 玄关（鞋柜 + 收纳箱）==============================
    P.append(put("cabinet", 0.85, 0, -3.6, math.pi / 2))
    P.append(put("box", -1.5, 0, -4.5, 0.2, "DYNAMIC"))

    # ================= F1 书房（东翼：书桌贴东墙 + 椅子 + 书架）================
    # 门洞 z[3.9,4.8] 在 x=1.2 墙南端，书桌迁到东墙 x≈4.7 让出门口动线
    P.append(put("desk", 4.68, 0, 3.00, -math.pi / 2))
    P.append(face("chair", 3.98, 0, 3.00, 4.68, 3.00))
    P.append(put("keyboard", 4.50, TABLE_TOP, 3.00, -math.pi / 2, "DYNAMIC"))
    P.append(put("book", 4.68, TABLE_TOP, 2.64, 0.3, "DYNAMIC"))
    P.append(put("mug", 4.74, TABLE_TOP, 3.32, 0, "DYNAMIC"))
    P.append(put("shelf", 1.4, 0, 1.5, math.pi / 2))
    P.append(put("book", 1.4, SHELF_TOPS[0], 1.32, 0.2, "DYNAMIC"))
    P.append(put("book", 1.4, SHELF_TOPS[1], 1.62, -0.2, "DYNAMIC"))
    P.append(put("book", 1.4, SHELF_TOPS[2], 1.45, 0.1, "DYNAMIC"))
    P.append(put("lamp", 3.3, 0, 4.55, 0, "DYNAMIC"))

    # ================= 电梯轿厢（停一楼，STATIC）============================
    P.append(put("elevator_car", 4.0, 0, -4.10, 0, "STATIC", rule="elevator"))

    # ================= F2 主卧（床贴西墙 + 双床头柜 + 衣柜）==================
    P.append(put("bed", -3.85, F2, 3.0, math.pi / 2))             # 床头板贴西墙
    # 火灾救援假人：躺在二楼主卧床面（床面 y≈F2+0.55），DYNAMIC 可搬运
    P.append(put("dummy", -3.85, F2 + 0.55, 3.0, math.pi / 2, "DYNAMIC", rule="rescue"))
    P.append(put("nightstand", -4.60, F2, 1.75, math.pi / 2))
    P.append(put("nightstand", -4.60, F2, 4.25, math.pi / 2))
    P.append(put("lamp", -4.60, F2 + NIGHT_TOP, 1.75, 0, "DYNAMIC"))
    P.append(put("mug", -4.60, F2 + NIGHT_TOP, 4.22, 0, "DYNAMIC"))
    P.append(put("book", -4.58, F2 + NIGHT_TOP, 4.32, 0.5, "DYNAMIC"))
    P.append(put("cabinet", -2.5, F2, 0.55))                      # 衣柜贴北墙 z=0.2
    P.append(put("box", -1.35, F2, 4.50, 0.2, "DYNAMIC"))
    P.append(put("box", -1.72, F2, 4.38, -0.3, "DYNAMIC"))

    # ================= F2 次卧（床贴北墙 + 床头柜 + 衣柜）====================
    P.append(put("bed", -3.9, F2, -3.85, 0))                      # 床头板贴北墙
    P.append(put("nightstand", -2.48, F2, -4.60, 0))
    P.append(put("book", -2.48, F2 + NIGHT_TOP, -4.58, 0.3, "DYNAMIC"))
    P.append(put("cabinet", -2.35, F2, -1.65, math.pi / 2))

    # ================= F2 书房（书桌贴北墙 + 书架）==========================
    P.append(put("desk", -0.3, F2, -4.50))
    P.append(face("chair", -0.3, F2, -3.85, -0.3, -4.50))
    P.append(put("keyboard", -0.3, F2 + TABLE_TOP, -4.36, 0, "DYNAMIC"))
    P.append(put("book", -0.62, F2 + TABLE_TOP, -4.50, 0.2, "DYNAMIC"))
    P.append(put("cup", 0.02, F2 + TABLE_TOP, -4.42, 0, "DYNAMIC"))
    # 书架贴次卧/书房隔墙 x=-2（原位置在楼梯间洞口上方，会悬空）
    P.append(put("shelf", -1.75, F2, -3.0, math.pi / 2))
    P.append(put("book", -1.75, F2 + SHELF_TOPS[0], -3.12, 0.1, "DYNAMIC"))
    P.append(put("book", -1.75, F2 + SHELF_TOPS[1], -2.88, -0.1, "DYNAMIC"))

    # ================= F2 茶室（桌 + 2 椅 + 边柜）===========================
    # 门洞 z[2.5,3.4] 在 x=1.2 墙。茶室仅 10㎡，human(r0.3) erosion 后通道紧张，
    # 桌子贴东墙放，门口->室内留 1.7m 主通道；椅子围桌但不挡动线。
    P.append(put("table", 4.30, F2, 3.20))                        # 桌贴东墙
    P.append(face("chair", 4.30, F2, 4.20, 4.30, 3.20))           # 桌北
    P.append(face("chair", 3.50, F2, 2.20, 4.30, 3.20))           # 桌西南
    P.append(put("cup", 4.10, F2 + TABLE_TOP, 3.08, 0, "DYNAMIC"))
    P.append(put("cup", 4.50, F2 + TABLE_TOP, 3.32, 0, "DYNAMIC"))
    P.append(put("bottle", 4.30, F2 + TABLE_TOP, 3.20, 0, "DYNAMIC"))
    P.append(put("cabinet", 1.60, F2, 4.55))                      # 边柜贴南墙西
    P.append(put("lamp", 4.70, F2, 4.60, 0, "DYNAMIC"))           # 落地灯东南角

    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    placements = build_plan(rng)
    plan = {
        "format": "habitat_scene_builder_v2_plan",
        "stage": "demo_assets/stages/villa_2f.glb",
        "scene": "villa_000",
        "note": "两层别墅(100m^2/层) 按真实户型工程数据布局；"
                "可手工编辑后用 --plan-in 重建。电梯轿厢 elevator_car 停在一楼。",
        "placements": placements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    n_dyn = sum(1 for p in placements if p["motion_type"] == "DYNAMIC")
    print(f"{args.out}  共 {len(placements)} 件（DYNAMIC 可交互 {n_dyn} 件）")


if __name__ == "__main__":
    main()
