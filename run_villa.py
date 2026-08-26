#!/usr/bin/env python3
"""
Run a robot navigation demo in a procedurally generated villa scene.

Usage:
    /data/hsb/env/bin/python run_villa.py --num_floors 3 --record
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import magnum as mn

# Add habitat-sim to path if not installed
HSB_PYTHON_PATH = "/data/hsb/build/habitat-sim/src_python"
HSB_DATA_PATH = "/data/hsb/build/habitat-sim/data"
if os.path.exists(HSB_PYTHON_PATH) and HSB_PYTHON_PATH not in sys.path:
    sys.path.insert(0, HSB_PYTHON_PATH)

import habitat_sim
from habitat_sim.utils import viz_utils as vut

from villa_generator import VillaBuilder, VillaConfig, FLOOR_HEIGHT


def make_cfg(settings):
    """Create simulator configuration."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    sim_cfg.allow_sliding = True
    sim_cfg.physics_config_file = os.path.join(HSB_DATA_PATH, "default.physics_config.json")
    sim_cfg.random_seed = settings.get("seed", 42)

    # Sensor specifications
    sensor_specs = []

    # First-person RGB camera
    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "rgba_camera"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [settings["height"], settings["width"]]
    color_spec.position = [0.0, 1.2, 0.0]
    color_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_spec.hfov = 90
    sensor_specs.append(color_spec)

    # Depth camera
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_camera"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [settings["height"], settings["width"]]
    depth_spec.position = [0.0, 1.2, 0.0]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_spec.hfov = 90
    sensor_specs.append(depth_spec)

    # Third-person follow camera
    third_spec = habitat_sim.CameraSensorSpec()
    third_spec.uuid = "third_person"
    third_spec.sensor_type = habitat_sim.SensorType.COLOR
    third_spec.resolution = [settings["height"], settings["width"]]
    third_spec.position = [0.0, 2.5, -3.5]
    third_spec.orientation = [-30.0, 0.0, 0.0]
    third_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    third_spec.hfov = 90
    sensor_specs.append(third_spec)

    # Agent configuration (cylinder collision body for robot)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.2
    agent_cfg.radius = 0.3
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up",
            habitat_sim.agent.ActuationSpec(amount=10.0),
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down",
            habitat_sim.agent.ActuationSpec(amount=10.0),
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def place_robot(sim, builder, position):
    """Place the agent at the spawn position. The agent has a built-in
    cylinder collision body (radius=0.3, height=1.5) for physics collision.
    Returns None since we use the default agent body.
    """
    # Agent body is configured in AgentConfiguration (cylinder collider)
    # No separate visual robot body needed - the camera IS the robot
    return None


def navigate_to_waypoint(sim, robot, target, dt=1.0 / 60.0, max_steps=600):
    """Navigate robot toward a target waypoint with collision avoidance.

    Yields sensor observations at each step.
    """
    agent = sim.agents[0]
    target_pos = np.array(target, dtype=np.float32)
    steps = 0
    stuck_counter = 0
    last_pos = None

    while steps < max_steps:
        state = agent.get_state()
        pos = np.array(state.position, dtype=np.float32)

        # Check distance to target (ignore Y for horizontal navigation)
        dx = target_pos[0] - pos[0]
        dz = target_pos[2] - pos[2]
        dist = math.sqrt(dx * dx + dz * dz)

        if dist < 0.5:
            break

        # Check if stuck
        if last_pos is not None:
            moved = np.linalg.norm(pos - last_pos)
            if moved < 0.01:
                stuck_counter += 1
                if stuck_counter > 30:
                    # Stuck, try turning
                    agent.act("turn_left")
                    stuck_counter = 0
            else:
                stuck_counter = 0
        last_pos = pos.copy()

        # Compute desired yaw to face target
        desired_yaw = math.atan2(-dx, -dz)
        current_yaw = state.rotation
        # Extract yaw from quaternion
        q = state.rotation
        if hasattr(q, 'vector'):
            qx, qy, qz, qw = q.vector[0], q.vector[1], q.vector[2], q.scalar
        else:
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Yaw from quaternion
        siny_cosp = 2 * (qw * qy + qx * qz)
        cosy_cosp = 1 - 2 * (qy * qy + qx * qx)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Angle difference
        angle_diff = desired_yaw - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Turn or move
        if abs(angle_diff) > 0.2:
            if angle_diff > 0:
                agent.act("turn_left")
            else:
                agent.act("turn_right")
        else:
            agent.act("move_forward")

        # Step physics (agent collision is handled by habitat-sim)
        sim.step_physics(dt)

        # Get observations
        observations = sim.get_sensor_observations()
        yield observations

        steps += 1


def run_villa_demo(num_floors=2, seed=42, record=True, output_dir="output",
                   width=640, height=480, fps=30):
    """Main entry point: build villa and run navigation demo."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Villa Scene Demo ===")
    print(f"Floors: {num_floors}, Seed: {seed}")
    print(f"Resolution: {width}x{height}")

    settings = {
        "width": width,
        "height": height,
        "seed": seed,
    }

    cfg = make_cfg(settings)
    sim = habitat_sim.Simulator(cfg)

    try:
        # Build the villa
        print("\n[1/4] Building villa scene...")
        t0 = time.time()
        config = VillaConfig(num_floors=num_floors, seed=seed)
        builder = VillaBuilder(sim, config)
        builder.build()
        print(f"  Built in {time.time()-t0:.1f}s")

        # Initialize agent at spawn point
        print("[2/4] Placing robot...")
        spawn_points = builder.get_spawn_points()
        start_pos = list(spawn_points[0])
        start_pos[1] += 0.05  # slight offset above floor

        agent_state = habitat_sim.AgentState()
        agent_state.position = start_pos
        sim.initialize_agent(0, agent_state)

        robot = place_robot(sim, builder, start_pos)

        # Set up lighting
        lights = []
        # Sun/directional light through windows
        sun = habitat_sim.gfx.LightInfo()
        sun.vector = mn.Vector4(0.5, -0.8, -0.3, 0)  # directional (w=0)
        sun.color = mn.Color3(1.0, 0.95, 0.9)
        sun.model = habitat_sim.gfx.LightPositionModel.Global
        lights.append(sun)

        # Interior ambient point lights on each floor
        for floor in range(num_floors):
            yf = floor * FLOOR_HEIGHT + 2.8
            for cx, cz in [(0, 0), (4, -3), (4, 3), (-4, 3)]:
                pl = habitat_sim.gfx.LightInfo()
                pl.vector = mn.Vector4(cx, yf, cz, 1)  # point (w=1)
                pl.color = mn.Color3(0.9, 0.85, 0.75)
                pl.model = habitat_sim.gfx.LightPositionModel.Global
                lights.append(pl)

        sim.set_light_setup(lights, "villa_lights")

        # Let physics settle
        for _ in range(30):
            sim.step_physics(1.0 / 60.0)

        # Navigate through the villa
        print("[3/4] Running navigation...")
        waypoints = builder.get_navigation_waypoints()
        print(f"  Total waypoints: {len(waypoints)}")

        all_observations = []
        total_frames = 0

        for i, wp in enumerate(waypoints):
            print(f"  Navigating to waypoint {i+1}/{len(waypoints)}: "
                  f"({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")

            # Elevator call if waypoint is on different floor
            if config.include_elevator and i == 0:
                # Keep elevator at ground floor
                builder.update_elevator(0.1, target_floor=0)

            for obs in navigate_to_waypoint(sim, robot, wp, max_steps=400):
                all_observations.append(obs)
                total_frames += 1

                # Animate elevator: cycle between floors during navigation
                if config.include_elevator:
                    elev_target = int((total_frames / 500)) % num_floors
                    builder.update_elevator(1.0 / 60.0, target_floor=elev_target)

        # Elevator demonstration: move elevator up and down at the end
        if config.include_elevator:
            print("  Demonstrating elevator...")
            for target_floor in range(num_floors):
                for _ in range(120):
                    builder.update_elevator(1.0 / 30.0, target_floor=target_floor)
                    sim.step_physics(1.0 / 60.0)
                    all_observations.append(sim.get_sensor_observations())
                    total_frames += 1

        # Add some idle frames for viewing
        for _ in range(60):
            sim.step_physics(1.0 / 60.0)
            all_observations.append(sim.get_sensor_observations())

        print(f"  Total frames recorded: {total_frames}")

        # Save video
        if record:
            print("[4/4] Saving video...")
            video_path = os.path.join(
                output_dir,
                f"villa_{num_floors}floor_seed{seed}.mp4"
            )
            vut.make_video(
                all_observations,
                "rgba_camera",
                "color",
                video_path.replace(".mp4", ""),
                fps=fps,
                open_vid=False,
            )
            print(f"  First-person video saved: {video_path}")

            # Third-person view
            tp_path = os.path.join(
                output_dir,
                f"villa_{num_floors}floor_seed{seed}_3rdperson.mp4"
            )
            vut.make_video(
                all_observations,
                "third_person",
                "color",
                tp_path.replace(".mp4", ""),
                fps=fps,
                open_vid=False,
            )
            print(f"  Third-person video saved: {tp_path}")

        print("\n=== Demo complete ===")
        return all_observations

    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(
        description="Procedural villa navigation demo in Habitat-Sim"
    )
    parser.add_argument("--num_floors", type=int, default=None,
                        help="Number of floors (2-4, random if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--no_record", action="store_true",
                        help="Do not save video")
    args = parser.parse_args()

    # Random number of floors if not specified
    if args.num_floors is None:
        random.seed(args.seed)
        args.num_floors = random.randint(2, 4)
        print(f"Randomly selected {args.num_floors} floors")

    if args.num_floors < 2 or args.num_floors > 4:
        print("Error: num_floors must be between 2 and 4")
        sys.exit(1)

    run_villa_demo(
        num_floors=args.num_floors,
        seed=args.seed,
        record=not args.no_record,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
