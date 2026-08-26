#!/usr/bin/env python3
"""
Batch-generate procedural villa scenes for downstream model training.

Output structure:
    dataset/
      scenes/
        scene_0000/
          scene.glb          - 3D geometry (glTF binary)
          metadata.json      - rooms, spawn points, waypoints, config
          preview.png        - perspective preview render
          topdown.png        - top-down floor plan
        scene_0001/
        ...
      index.json             - global index of all scenes
      README.txt             - dataset description

Usage:
    /data/hsb/env/bin/python batch_generate.py --num_scenes 40 --output_dir dataset
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HSB_PYTHON_PATH = "/data/hsb/build/habitat-sim/src_python"
HSB_DATA_PATH = "/data/hsb/build/habitat-sim/data"
if os.path.exists(HSB_PYTHON_PATH) and HSB_PYTHON_PATH not in sys.path:
    sys.path.insert(0, HSB_PYTHON_PATH)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import magnum as mn
import habitat_sim
from habitat_sim.utils import viz_utils as vut

from villa_generator import VillaBuilder, VillaConfig, COLORS


def make_sim_cfg(width=480, height=480):
    """Create a headless simulator config with preview cameras."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    sim_cfg.physics_config_file = os.path.join(
        HSB_DATA_PATH, "default.physics_config.json"
    )

    sensor_specs = []

    # Perspective preview camera
    persp = habitat_sim.CameraSensorSpec()
    persp.uuid = "preview"
    persp.sensor_type = habitat_sim.SensorType.COLOR
    persp.resolution = [height, width]
    persp.position = [0.0, 8.0, -12.0]
    persp.orientation = [-35.0, 0.0, 0.0]
    persp.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    persp.hfov = 75
    sensor_specs.append(persp)

    # Top-down camera
    top = habitat_sim.CameraSensorSpec()
    top.uuid = "topdown"
    top.sensor_type = habitat_sim.SensorType.COLOR
    top.resolution = [height, width]
    top.position = [0.0, 20.0, 0.01]
    top.orientation = [-90.0, 0.0, 0.0]
    top.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    top.hfov = 90
    sensor_specs.append(top)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.2
    agent_cfg.radius = 0.3
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def render_preview(sim, builder):
    """Render perspective and top-down images. Returns (preview, topdown)."""
    # Position preview camera to frame the villa
    cfg = builder.config
    max_dim = max(cfg.villa_width, cfg.villa_depth)
    cam_dist = max_dim * 0.9
    cam_height = max_dim * 0.55

    # Set preview camera transform
    agent = sim.agents[0]
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(0, cam_height, -cam_dist)
    # Quaternion [x, y, z, w] for -35 degree pitch around X axis
    pitch = np.radians(-35)
    state.rotation = [float(np.sin(pitch/2)), 0.0, 0.0, float(np.cos(pitch/2))]
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    preview = obs["preview"]
    topdown = obs["topdown"]
    return preview, topdown


def generate_scene(sim, scene_idx, output_dir, seed):
    """Generate a single scene and save all assets."""
    scene_name = f"scene_{scene_idx:04d}"
    scene_dir = os.path.join(output_dir, "scenes", scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # Randomize config
    vc = VillaConfig.random(seed)
    builder = VillaBuilder(sim, vc)
    builder.build()

    # Step physics to settle
    for _ in range(10):
        sim.step_physics(1.0 / 60.0)

    # Export GLB
    glb_path = os.path.join(scene_dir, "scene.glb")
    builder.export_glb(glb_path)

    # Export metadata
    metadata = builder.get_metadata()
    metadata["scene_id"] = scene_name
    metadata["glb_file"] = "scene.glb"
    meta_path = os.path.join(scene_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Render previews
    preview, topdown = render_preview(sim, builder)
    import imageio
    imageio.imwrite(os.path.join(scene_dir, "preview.png"),
                    np.asarray(preview)[..., :3])
    imageio.imwrite(os.path.join(scene_dir, "topdown.png"),
                    np.asarray(topdown)[..., :3])

    # Clean up all objects for next scene
    sim.get_rigid_object_manager().remove_all_objects()

    return {
        "scene_id": scene_name,
        "seed": seed,
        "num_floors": vc.num_floors,
        "villa_width": round(vc.villa_width, 2),
        "villa_depth": round(vc.villa_depth, 2),
        "floor_height": round(vc.floor_height, 2),
        "has_elevator": vc.include_elevator,
        "num_objects": metadata["num_objects"],
        "num_rooms": len(metadata["rooms"]),
        "glb_file": "scene.glb",
        "preview": "preview.png",
        "topdown": "topdown.png",
        "glb_size_mb": round(os.path.getsize(glb_path) / 1024 / 1024, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate villa scenes for training"
    )
    parser.add_argument("--num_scenes", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scenes_dir = os.path.join(args.output_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    print(f"=== Batch Scene Generation ===")
    print(f"Scenes: {args.num_scenes}, Output: {args.output_dir}")
    print(f"Seed range: {args.seed_start} - {args.seed_start + args.num_scenes - 1}")

    cfg = make_sim_cfg()
    sim = habitat_sim.Simulator(cfg)

    index = []
    t_start = time.time()

    try:
        for i in range(args.num_scenes):
            seed = args.seed_start + i
            t0 = time.time()
            info = generate_scene(sim, i, args.output_dir, seed)
            dt = time.time() - t0
            index.append(info)
            print(
                f"  [{i+1:3d}/{args.num_scenes}] {info['scene_id']} "
                f"({info['num_floors']}F, {info['num_objects']} objs, "
                f"{info['glb_size_mb']}MB, {dt:.1f}s)"
            )
    finally:
        sim.close()

    # Save global index
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump({
            "dataset": "habitat_villa_scenes",
            "version": "1.0",
            "num_scenes": len(index),
            "generator": "habitat_scene_builder",
            "scenes": index,
        }, f, indent=2)

    # Save README
    readme_path = os.path.join(args.output_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("Habitat Villa Scene Dataset\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total scenes: {len(index)}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Each scene directory contains:\n")
        f.write("  scene.glb      - 3D geometry (glTF 2.0 binary, Y-up, meters)\n")
        f.write("  metadata.json  - rooms, spawn points, navigation waypoints\n")
        f.write("  preview.png    - perspective render\n")
        f.write("  topdown.png    - top-down floor plan\n\n")
        f.write("metadata.json fields:\n")
        f.write("  seed, num_floors, floor_height, villa_width, villa_depth\n")
        f.write("  has_elevator, num_objects\n")
        f.write("  rooms[]: name, type, floor, center[x,z], size[w,d]\n")
        f.write("  spawn_points[]: [x,y,z] per floor\n")
        f.write("  navigation_waypoints[]: scripted path through all floors\n")
        f.write("\nScene variety: 2-4 floors, random footprint (14-20m x 10-15m),\n")
        f.write("random floor height (2.9-3.5m), optional elevator, U-staircase,\n")
        f.write("furnished rooms (living, kitchen, dining, bedroom, bathroom, foyer).\n")
        f.write("All objects are primitive collision shapes (box/cylinder) suitable\n")
        f.write("for Bullet physics. Load in Habitat-Sim with scene_id = scene.glb.\n")

    total_time = time.time() - t_start
    total_size_mb = sum(s["glb_size_mb"] for s in index)
    print(f"\n=== Generation complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/args.num_scenes:.1f}s/scene)")
    print(f"Total GLB size: {total_size_mb:.1f}MB")
    print(f"Index saved: {index_path}")


if __name__ == "__main__":
    main()
