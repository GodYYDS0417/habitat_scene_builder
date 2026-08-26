"""
Procedural multi-story villa scene generator for Habitat-Sim 0.3.1.

Generates 2-4 story villas with:
- Realistic room layouts (living room, kitchen, bedrooms, bathrooms, etc.)
- Villa staircase connecting all floors
- Functional elevator (kinematic platform)
- Physical collision on all objects (no clipping)
- Realistic furniture placement
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

try:
    import magnum as mn
except ImportError:
    mn = None

import habitat_sim


def _v3(x, y, z):
    """Create a magnum Vector3 from coordinates."""
    if mn is not None:
        return mn.Vector3(x, y, z)
    return (x, y, z)


# ============================================================================
# Constants (metric units, 1 unit = 1 meter)
# ============================================================================

FLOOR_HEIGHT = 3.2
WALL_THICKNESS = 0.18
SLAB_THICKNESS = 0.2
DOOR_WIDTH = 0.9
DOOR_HEIGHT = 2.1
STAIR_RISER = 0.165
STAIR_TREAD = 0.28
STAIR_WIDTH = 1.2
ELEVATOR_WIDTH = 2.0
ELEVATOR_DEPTH = 2.2
ELEVATOR_DOOR_WIDTH = 1.2

# Villa footprint
VILLA_WIDTH = 16.0
VILLA_DEPTH = 12.0

# Material colors (R, G, B, A) in linear sRGB
COLORS = {
    "wall": (0.92, 0.91, 0.88, 1.0),
    "wall_accent": (0.82, 0.78, 0.72, 1.0),
    "floor_wood": (0.55, 0.40, 0.25, 1.0),
    "floor_marble": (0.85, 0.83, 0.80, 1.0),
    "floor_carpet": (0.45, 0.38, 0.35, 1.0),
    "stair_wood": (0.50, 0.36, 0.22, 1.0),
    "ceiling": (0.95, 0.95, 0.95, 1.0),
    "sofa": (0.35, 0.35, 0.38, 1.0),
    "sofa_cushion": (0.45, 0.45, 0.48, 1.0),
    "bed_frame": (0.40, 0.30, 0.22, 1.0),
    "mattress": (0.95, 0.95, 0.93, 1.0),
    "pillow": (1.0, 1.0, 0.98, 1.0),
    "table_wood": (0.48, 0.35, 0.22, 1.0),
    "chair_wood": (0.50, 0.37, 0.24, 1.0),
    "cabinet_wood": (0.42, 0.30, 0.18, 1.0),
    "countertop": (0.60, 0.58, 0.55, 1.0),
    "appliance": (0.75, 0.75, 0.77, 1.0),
    "appliance_dark": (0.25, 0.25, 0.27, 1.0),
    "sink": (0.80, 0.82, 0.85, 1.0),
    "bathtub": (0.90, 0.92, 0.94, 1.0),
    "toilet": (0.95, 0.95, 0.95, 1.0),
    "rug": (0.55, 0.25, 0.25, 1.0),
    "tv": (0.08, 0.08, 0.10, 1.0),
    "bookshelf": (0.40, 0.28, 0.18, 1.0),
    "lamp_base": (0.30, 0.30, 0.32, 1.0),
    "lamp_shade": (0.95, 0.90, 0.75, 1.0),
    "plant_pot": (0.45, 0.32, 0.22, 1.0),
    "plant_green": (0.25, 0.45, 0.20, 1.0),
    "elevator_metal": (0.65, 0.67, 0.70, 1.0),
    "elevator_door": (0.55, 0.57, 0.60, 1.0),
    "railing": (0.30, 0.30, 0.32, 1.0),
    "door_wood": (0.45, 0.32, 0.20, 1.0),
    "window_glass": (0.70, 0.82, 0.90, 0.4),
    "frame_white": (0.95, 0.95, 0.95, 1.0),
    "robot_body": (0.20, 0.45, 0.80, 1.0),
    "robot_accent": (0.15, 0.15, 0.18, 1.0),
}


@dataclass
class Room:
    """A room in the villa."""
    name: str
    x: float  # center X
    z: float  # center Z
    width: float  # X extent
    depth: float  # Z extent
    floor: int
    room_type: str = "generic"
    color: Tuple[float, ...] = COLORS["floor_wood"]


@dataclass
class VillaConfig:
    """Configuration for procedural villa generation."""
    num_floors: int = 2
    seed: int = 42
    villa_width: float = VILLA_WIDTH
    villa_depth: float = VILLA_DEPTH
    floor_height: float = FLOOR_HEIGHT
    wall_thickness: float = WALL_THICKNESS
    include_elevator: bool = True
    include_basement: bool = False
    furniture_density: float = 1.0


class VillaBuilder:
    """Builds a complete villa scene in Habitat-Sim."""

    def __init__(self, sim: habitat_sim.Simulator, config: VillaConfig = None):
        self.sim = sim
        self.config = config or VillaConfig()
        self.rng = random.Random(self.config.seed)
        np.random.seed(self.config.seed)

        self.obj_mgr = sim.get_object_template_manager()
        self.rigid_mgr = sim.get_rigid_object_manager()
        self.asset_mgr = sim.get_asset_template_manager()

        self.rooms: List[Room] = []
        self.objects: Dict[str, int] = {}
        self.elevator_platform = None
        self.elevator_target_floor = 0
        self.elevator_direction = 0

        # Template cache
        self._template_cache: Dict[str, int] = {}

    def build(self):
        """Build the complete villa scene."""
        self._register_primitive_templates()
        self._build_slab_and_ceiling()
        self._build_rooms()
        self._build_staircase()
        if self.config.include_elevator:
            self._build_elevator()
        self._build_furniture()
        self._build_exterior_details()

    # ------------------------------------------------------------------
    # Template registration
    # ------------------------------------------------------------------

    def _register_primitive_templates(self):
        """Register object templates for all primitive shapes we'll use.

        Habitat-sim provides built-in primitive templates (e.g. "cubeSolid")
        that have use_mesh_collision=false, enabling Bullet primitive colliders.
        """
        # Built-in primitive template handles
        self._cube_handle = "cubeSolid"
        self._cylinder_handle = (
            "cylinderSolid_rings_1_segments_12_halfLen_1"
            "_useTexCoords_false_useTangents_false_capEnds_true"
        )
        self._cone_handle = (
            "coneSolid_segments_12_halfLen_1.25_rings_1"
            "_useTexCoords_false_useTangents_false_capEnd_true"
        )
        self._sphere_handle = "icosphereSolid_subdivs_1"

        # Verify templates exist
        for h in [self._cube_handle, self._cylinder_handle,
                  self._cone_handle, self._sphere_handle]:
            if self.obj_mgr.get_template_by_handle(h) is None:
                raise RuntimeError(f"Primitive template not found: {h}")

    def _get_box_template(self, name: str, scale: Tuple[float, float, float],
                          color_key: str = "wall") -> int:
        """Get or create a box template with specific scale.

        Uses the built-in cubeSolid primitive which has primitive collision
        (no mesh collision needed).
        """
        key = f"box_{name}_{scale[0]:.3f}_{scale[1]:.3f}_{scale[2]:.3f}"
        if key in self._template_cache:
            return self._template_cache[key]

        # Get a copy of the built-in cubeSolid template
        template = self.obj_mgr.get_template_by_handle("cubeSolid")
        if template is None:
            raise RuntimeError("Built-in cubeSolid template not found")
        # Cube primitive mesh is 2x2x2 (vertices at ±1), so divide by 2.
        template.scale = [scale[0] / 2.0, scale[1] / 2.0, scale[2] / 2.0]
        template.margin = 0.005
        template.friction_coefficient = 0.6
        template.restitution_coefficient = 0.05

        tid = self.obj_mgr.register_template(template, key)
        self._template_cache[key] = tid
        return tid

    def _get_cylinder_template(self, name: str, radius: float, height: float,
                               color_key: str = "wall") -> int:
        """Get or create a cylinder template."""
        key = f"cyl_{name}_{radius:.4f}_{height:.4f}"
        if key in self._template_cache:
            return self._template_cache[key]

        template = self.obj_mgr.get_template_by_handle(self._cylinder_handle)
        if template is None:
            raise RuntimeError("Built-in cylinder template not found")
        # Cylinder primitive: radius=1, height=2*halfLen=2
        template.scale = [radius, height / 2.0, radius]
        template.margin = 0.005
        template.friction_coefficient = 0.5

        tid = self.obj_mgr.register_template(template, key)
        self._template_cache[key] = tid
        return tid

    # ------------------------------------------------------------------
    # Object placement helpers
    # ------------------------------------------------------------------

    def _add_static_box(self, name: str,
                        position: Tuple[float, float, float],
                        size: Tuple[float, float, float],
                        color_key: str = "wall",
                        rotation: Tuple[float, float, float] = None) -> int:
        """Add a static box object. Returns object ID."""
        tid = self._get_box_template(name, size, color_key)
        obj = self.rigid_mgr.add_object_by_template_id(tid)
        obj.translation = _v3(*position)
        if rotation is not None and mn is not None:
            q = (mn.Quaternion.rotation(mn.Rad(rotation[1]), mn.Vector3.y_axis())
                 * mn.Quaternion.rotation(mn.Rad(rotation[0]), mn.Vector3.x_axis())
                 * mn.Quaternion.rotation(mn.Rad(rotation[2]), mn.Vector3.z_axis()))
            obj.rotation = q
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
        return obj.object_id

    def _add_static_cylinder(self, name: str,
                             position: Tuple[float, float, float],
                             radius: float, height: float,
                             color_key: str = "railing") -> int:
        """Add a static cylinder object."""
        tid = self._get_cylinder_template(name, radius, height, color_key)
        obj = self.rigid_mgr.add_object_by_template_id(tid)
        obj.translation = _v3(*position)
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
        return obj.object_id

    def _add_box(self, name: str,
                 position: Tuple[float, float, float],
                 size: Tuple[float, float, float],
                 color_key: str = "wall",
                 motion_type: str = "static",
                 mass: float = 0.0) -> int:
        """Add a box with specified motion type."""
        tid = self._get_box_template(name, size, color_key)
        obj = self.rigid_mgr.add_object_by_template_id(tid)
        obj.translation = _v3(*position)
        if motion_type == "static":
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
        elif motion_type == "kinematic":
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        else:
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj.mass = mass
        return obj.object_id

    # ------------------------------------------------------------------
    # Floor slabs and ceilings
    # ------------------------------------------------------------------

    def _build_slab_and_ceiling(self):
        """Build floor slabs and ceilings for all floors."""
        for floor in range(self.config.num_floors):
            y_base = floor * self.config.floor_height

            # Floor slab
            self._add_static_box(
                f"floor_slab_{floor}",
                (0, y_base - SLAB_THICKNESS / 2, 0),
                (self.config.villa_width + 1.0, SLAB_THICKNESS,
                 self.config.villa_depth + 1.0),
                "floor_marble" if floor == 0 else "floor_wood",
            )

            # Ceiling (except for top floor, which gets a roof)
            if floor < self.config.num_floors - 1:
                self._add_static_box(
                    f"ceiling_{floor}",
                    (0, y_base + self.config.floor_height - 0.05, 0),
                    (self.config.villa_width + 0.5, 0.1,
                     self.config.villa_depth + 0.5),
                    "ceiling",
                )
            else:
                # Roof slab
                self._add_static_box(
                    f"roof_{floor}",
                    (0, y_base + self.config.floor_height, 0),
                    (self.config.villa_width + 1.5, SLAB_THICKNESS,
                     self.config.villa_depth + 1.5),
                    "wall_accent",
                )

    # ------------------------------------------------------------------
    # Room layout
    # ------------------------------------------------------------------

    def _build_rooms(self):
        """Build interior and exterior walls for all floors."""
        hw = self.config.villa_width / 2
        hd = self.config.villa_depth / 2
        wt = self.config.wall_thickness

        # Floor 0: Living room, kitchen, dining, bathroom, staircase
        # Floor 1+: Bedrooms, bathroom, study
        for floor in range(self.config.num_floors):
            y_base = floor * self.config.floor_height
            wall_h = self.config.floor_height - SLAB_THICKNESS
            y_center = y_base + wall_h / 2

            # Exterior walls
            # Front wall (z = -hd) with windows
            self._build_wall_with_openings(
                "front", floor, y_center, wall_h,
                position_z=-hd, is_x_wall=True,
                length=self.config.villa_width, thickness=wt,
                openings=[(-3, 1.5), (3, 1.5)] if floor > 0 else [(-2, 1.2), (4, 1.8)],
            )
            # Back wall (z = hd)
            self._build_wall_with_openings(
                "back", floor, y_center, wall_h,
                position_z=hd, is_x_wall=True,
                length=self.config.villa_width, thickness=wt,
                openings=[(-2, 1.2), (2, 1.2)],
            )
            # Left wall (x = -hw)
            self._build_wall_with_openings(
                "left", floor, y_center, wall_h,
                position_x=-hw, is_x_wall=False,
                length=self.config.villa_depth, thickness=wt,
                openings=[],
            )
            # Right wall (x = hw)
            self._build_wall_with_openings(
                "right", floor, y_center, wall_h,
                position_x=hw, is_x_wall=False,
                length=self.config.villa_depth, thickness=wt,
                openings=[],
            )

            if floor == 0:
                self._build_ground_floor_walls(y_base, wall_h)
            else:
                self._build_upper_floor_walls(floor, y_base, wall_h)

    def _build_wall_with_openings(self, name: str, floor: int, y_center: float,
                                   wall_h: float, position_z: float = None,
                                   position_x: float = None,
                                   is_x_wall: bool = True,
                                   length: float = 10.0,
                                   thickness: float = 0.18,
                                   openings: List[Tuple[float, float]] = None):
        """Build a wall with openings (doors/windows).

        openings: list of (center_coord, width) along the wall length.
        For windows, y position is at 1.0m height. For doors, at floor level.
        """
        openings = openings or []
        # Sort openings by position
        openings_sorted = sorted(openings, key=lambda o: o[0])

        # Build wall segments between openings
        prev_edge = -length / 2
        segments = []

        for center, width in openings_sorted:
            left = center - width / 2
            right = center + width / 2
            if left > prev_edge:
                segments.append((prev_edge, left))
            # Add lintel above the opening (for doors/windows that don't reach ceiling)
            lintel_h = wall_h - DOOR_HEIGHT
            if lintel_h > 0.05:
                lintel_y = y_center + DOOR_HEIGHT / 2 - lintel_h / 2 + 0.1
                if is_x_wall:
                    self._add_static_box(
                        f"wall_{name}_lintel_{floor}_{center:.1f}",
                        (center, lintel_y, position_z),
                        (width, lintel_h, thickness),
                    )
                else:
                    self._add_static_box(
                        f"wall_{name}_lintel_{floor}_{center:.1f}",
                        (position_x, lintel_y, center),
                        (thickness, lintel_h, width),
                    )
            prev_edge = right

        if prev_edge < length / 2:
            segments.append((prev_edge, length / 2))

        for i, (s, e) in enumerate(segments):
            seg_len = e - s
            if seg_len < 0.05:
                continue
            center_coord = (s + e) / 2
            if is_x_wall:
                self._add_static_box(
                    f"wall_{name}_{floor}_{i}",
                    (center_coord, y_center, position_z),
                    (seg_len, wall_h, thickness),
                )
            else:
                self._add_static_box(
                    f"wall_{name}_{floor}_{i}",
                    (position_x, y_center, center_coord),
                    (thickness, wall_h, seg_len),
                )

        # Add window sills and glass
        for center, width in openings_sorted:
            sill_y = 1.0 + 0.04  # window at 1m height
            if is_x_wall:
                # Glass pane
                self._add_static_box(
                    f"window_{name}_{floor}_{center:.1f}",
                    (center, y_center + 0.3, position_z),
                    (width, 1.4, 0.02),
                    "window_glass",
                )
            else:
                self._add_static_box(
                    f"window_{name}_{floor}_{center:.1f}",
                    (position_x, y_center + 0.3, center),
                    (0.02, 1.4, width),
                    "window_glass",
                )

    def _build_ground_floor_walls(self, y_base: float, wall_h: float):
        """Build ground floor interior walls: living room, kitchen, dining, bath."""
        y_center = y_base + wall_h / 2
        wt = self.config.wall_thickness
        hw = self.config.villa_width / 2
        hd = self.config.villa_depth / 2

        # Staircase area: x in [-7, -4.5], z in [-4, 0]
        # Elevator area: x in [-4, -2], z in [-4, -1.5]

        # Wall separating staircase/elevator from living room
        # Runs along z from -hd to hd at x = -4.5 (stair right wall)
        self._build_wall_with_openings(
            "stair_right", 0, y_center, wall_h,
            position_x=-4.5, is_x_wall=False,
            length=8.0, thickness=wt,
            openings=[(0, DOOR_WIDTH)],  # opening to living area
        )

        # Wall separating staircase from elevator at x = -3.25, z in [-4, -1.5]
        self._add_static_box(
            "wall_stair_elev_0",
            (-3.25, y_center, -2.75),
            (wt, wall_h, 2.5),
        )

        # Kitchen wall: z = 2.0, x in [-4.5, hw]
        self._build_wall_with_openings(
            "kitchen", 0, y_center, wall_h,
            position_z=2.0, is_x_wall=True,
            length=hw + 4.5, thickness=wt,
            openings=[(-1.5, DOOR_WIDTH), (3.0, DOOR_WIDTH)],
        )

        # Bathroom wall: x in [-4.5, -1.5], z in [2.0, 5.0]
        # Bathroom back wall
        self._add_static_box(
            "wall_bath_back_0",
            (-3.0, y_center, 5.0 - wt / 2),
            (3.0 + wt, wall_h, wt),
        )
        # Bathroom side wall (x = -1.5)
        self._add_static_box(
            "wall_bath_side_0",
            (-1.5 + wt / 2, y_center, 3.5),
            (wt, wall_h, 3.0),
        )

        # Dining area is between kitchen and living, no full wall

        # Register rooms
        self.rooms.append(Room("living_room", 2.0, -1.0, 11.0, 7.0, 0, "living"))
        self.rooms.append(Room("kitchen", 2.0, 4.0, 8.0, 3.0, 0, "kitchen"))
        self.rooms.append(Room("dining", 0.0, 1.0, 4.0, 2.0, 0, "dining"))
        self.rooms.append(Room("bathroom_0", -3.0, 3.5, 3.0, 3.0, 0, "bathroom"))
        self.rooms.append(Room("foyer", -5.75, -2.0, 3.5, 4.0, 0, "foyer"))

    def _build_upper_floor_walls(self, floor: int, y_base: float, wall_h: float):
        """Build upper floor walls: bedrooms, bathroom, study."""
        y_center = y_base + wall_h / 2
        wt = self.config.wall_thickness
        hw = self.config.villa_width / 2

        # Corridor along the staircase/elevator side
        # Wall at x = -2.0 separating corridor from rooms
        self._build_wall_with_openings(
            f"corridor_{floor}", floor, y_center, wall_h,
            position_x=-2.0, is_x_wall=False,
            length=8.0, thickness=wt,
            openings=[(-2.5, DOOR_WIDTH), (1.5, DOOR_WIDTH)],
        )

        # Master bedroom: x in [0, hw], z in [-6, 0]
        # Wall between master bedroom and other room at z = 0
        self._build_wall_with_openings(
            f"bed_wall_{floor}", floor, y_center, wall_h,
            position_z=0.0, is_x_wall=True,
            length=hw + 2.0, thickness=wt,
            openings=[(3.0, DOOR_WIDTH)],
        )

        # Second bedroom / study: x in [-2, hw], z in [0, hd]
        # Bathroom: x in [-2, 1], z in [2, hd]
        # Bathroom walls
        self._add_static_box(
            f"wall_bath_back_{floor}",
            (-0.5, y_center, 5.0 - wt / 2),
            (3.0 + wt, wall_h, wt),
        )
        self._add_static_box(
            f"wall_bath_side_{floor}",
            (1.0 + wt / 2, y_center, 3.5),
            (wt, wall_h, 3.0),
        )

        # Register rooms
        self.rooms.append(Room(
            f"master_bedroom_{floor}", 4.0, -3.0, 10.0, 6.0, floor, "bedroom"))
        self.rooms.append(Room(
            f"bedroom_{floor}", 4.0, 2.5, 10.0, 5.0, floor, "bedroom"))
        self.rooms.append(Room(
            f"bathroom_{floor}", -0.5, 3.5, 3.0, 3.0, floor, "bathroom"))
        self.rooms.append(Room(
            f"corridor_{floor}", -3.25, -1.0, 2.5, 8.0, floor, "corridor"))

    # ------------------------------------------------------------------
    # Staircase
    # ------------------------------------------------------------------

    def _build_staircase(self):
        """Build a U-shaped villa staircase connecting all floors.

        Staircase occupies x: [-7, -4.5], z: [-5, 0] (3.5m x 5m)
        Two flights with a landing in between.
        """
        stair_x = -5.75  # center X of stairwell
        stair_w = STAIR_WIDTH

        for floor in range(self.config.num_floors - 1):
            y_base = floor * self.config.floor_height
            y_next = (floor + 1) * self.config.floor_height
            total_rise = self.config.floor_height
            num_steps = int(total_rise / STAIR_RISER)
            actual_riser = total_rise / num_steps

            # First flight: goes up along +z direction, from z=-4.5 to z=-1.5
            flight1_z_start = -4.5
            flight1_z_end = -1.5
            flight1_length = flight1_z_end - flight1_z_start
            steps_flight1 = num_steps // 2
            tread1 = flight1_length / steps_flight1

            for i in range(steps_flight1):
                step_y = y_base + (i + 1) * actual_riser - actual_riser / 2
                step_z = flight1_z_start + (i + 0.5) * tread1
                self._add_static_box(
                    f"stair_{floor}_f1_{i}",
                    (stair_x, step_y, step_z),
                    (stair_w, actual_riser, tread1 + 0.02),
                    "stair_wood",
                )

            # Landing at mid-height
            landing_y = y_base + steps_flight1 * actual_riser
            self._add_static_box(
                f"stair_landing_{floor}",
                (stair_x, landing_y - 0.06, -1.0),
                (stair_w + 1.5, 0.12, 1.2),
                "stair_wood",
            )

            # Second flight: goes up along -z direction, from z=-0.5 to z=-3.5
            # Offset in x by stair width to make U-shape
            flight2_x = stair_x + stair_w + 0.3
            flight2_z_start = -0.5
            flight2_z_end = -3.5
            flight2_length = abs(flight2_z_end - flight2_z_start)
            steps_flight2 = num_steps - steps_flight1
            tread2 = flight2_length / steps_flight2

            for i in range(steps_flight2):
                step_y = (y_base + steps_flight1 * actual_riser
                          + (i + 1) * actual_riser - actual_riser / 2)
                step_z = flight2_z_start - (i + 0.5) * tread2
                self._add_static_box(
                    f"stair_{floor}_f2_{i}",
                    (flight2_x, step_y, step_z),
                    (stair_w, actual_riser, tread2 + 0.02),
                    "stair_wood",
                )

            # Stair railings
            self._build_stair_railings(
                floor, stair_x, flight1_z_start, flight1_z_end,
                y_base, landing_y, steps_flight1, actual_riser, tread1, "up"
            )
            self._build_stair_railings(
                floor, flight2_x, flight2_z_start, flight2_z_end,
                landing_y, y_next, steps_flight2, actual_riser, tread2, "down"
            )

            # Handrail on landing
            self._add_static_box(
                f"stair_landing_rail_{floor}",
                (stair_x + 0.75, landing_y + 0.5, -1.0),
                (0.04, 0.04, 1.2),
                "railing",
            )

    def _build_stair_railings(self, floor, cx, z_start, z_end,
                               y_start, y_end, num_steps, riser, tread, direction):
        """Add railings along a stair flight."""
        for i in range(0, num_steps, 2):
            t = i / max(num_steps - 1, 1)
            y = y_start + t * (y_end - y_start) + 0.45
            if direction == "up":
                z = z_start + t * (z_end - z_start)
            else:
                z = z_start - t * (z_start - z_end)

            # Baluster
            self._add_static_cylinder(
                f"baluster_{floor}_{cx:.1f}_{i}",
                (cx - STAIR_WIDTH / 2, y, z),
                0.015, 0.9,
                "railing",
            )
            self._add_static_cylinder(
                f"baluster_r_{floor}_{cx:.1f}_{i}",
                (cx + STAIR_WIDTH / 2, y, z),
                0.015, 0.9,
                "railing",
            )

        # Handrail
        if direction == "up":
            z_mid = (z_start + z_end) / 2
            length = abs(z_end - z_start)
        else:
            z_mid = (z_start + z_end) / 2
            length = abs(z_end - z_start)

        y_mid = (y_start + y_end) / 2 + 0.9
        angle = math.atan2(y_end - y_start, length)

        # Simple handrail as a thin box
        self._add_static_box(
            f"handrail_l_{floor}_{cx:.1f}",
            (cx - STAIR_WIDTH / 2, y_mid, z_mid),
            (0.04, 0.04, length + 0.1),
            "railing",
            rotation=(angle, 0, 0) if mn is None else None,
        )
        self._add_static_box(
            f"handrail_r_{floor}_{cx:.1f}",
            (cx + STAIR_WIDTH / 2, y_mid, z_mid),
            (0.04, 0.04, length + 0.1),
            "railing",
            rotation=(angle, 0, 0) if mn is None else None,
        )

    # ------------------------------------------------------------------
    # Elevator
    # ------------------------------------------------------------------

    def _build_elevator(self):
        """Build elevator shaft and kinematic platform.

        Elevator occupies x: [-4, -2], z: [-4, -1.8] (2m x 2.2m)
        """
        ex = -3.0  # elevator center X
        ez = -2.9  # elevator center Z
        ew = ELEVATOR_WIDTH
        ed = ELEVATOR_DEPTH
        wt = self.config.wall_thickness
        total_h = self.config.num_floors * self.config.floor_height

        # Elevator shaft walls
        # Back wall
        self._add_static_box(
            "elev_wall_back",
            (ex, total_h / 2, ez - ed / 2),
            (ew + wt * 2, total_h, wt),
            "elevator_metal",
        )
        # Left wall
        self._add_static_box(
            "elev_wall_left",
            (ex - ew / 2, total_h / 2, ez),
            (wt, total_h, ed + wt * 2),
            "elevator_metal",
        )
        # Right wall (adjacent to stairs)
        self._add_static_box(
            "elev_wall_right",
            (ex + ew / 2, total_h / 2, ez),
            (wt, total_h, ed + wt * 2),
            "elevator_metal",
        )
        # Front wall with door openings on each floor
        for floor in range(self.config.num_floors):
            y_base = floor * self.config.floor_height
            # Wall segments around door
            door_w = ELEVATOR_DOOR_WIDTH
            side_w = (ew - door_w) / 2

            # Left of door
            self._add_static_box(
                f"elev_door_wall_l_{floor}",
                (ex - ew / 2 + side_w / 2,
                 y_base + self.config.floor_height / 2,
                 ez + ed / 2),
                (side_w, self.config.floor_height, wt),
                "elevator_metal",
            )
            # Right of door
            self._add_static_box(
                f"elev_door_wall_r_{floor}",
                (ex + ew / 2 - side_w / 2,
                 y_base + self.config.floor_height / 2,
                 ez + ed / 2),
                (side_w, self.config.floor_height, wt),
                "elevator_metal",
            )
            # Lintel
            self._add_static_box(
                f"elev_lintel_{floor}",
                (ex, y_base + DOOR_HEIGHT + (self.config.floor_height - DOOR_HEIGHT) / 2,
                 ez + ed / 2),
                (door_w, self.config.floor_height - DOOR_HEIGHT, wt),
                "elevator_metal",
            )

            # Door frame
            self._add_static_box(
                f"elev_frame_l_{floor}",
                (ex - door_w / 2 - 0.03, y_base + DOOR_HEIGHT / 2, ez + ed / 2 + 0.02),
                (0.06, DOOR_HEIGHT, 0.06),
                "elevator_door",
            )
            self._add_static_box(
                f"elev_frame_r_{floor}",
                (ex + door_w / 2 + 0.03, y_base + DOOR_HEIGHT / 2, ez + ed / 2 + 0.02),
                (0.06, DOOR_HEIGHT, 0.06),
                "elevator_door",
            )

        # Elevator platform (kinematic, can move)
        platform_tid = self._get_box_template(
            "elev_platform", (ew - 0.1, 0.08, ed - 0.1), "elevator_metal"
        )
        self.elevator_platform = self.rigid_mgr.add_object_by_template_id(platform_tid)
        self.elevator_platform.translation = _v3(ex, 0.04, ez)
        self.elevator_platform.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # Elevator ceiling on platform
        ceiling_tid = self._get_box_template(
            "elev_ceiling", (ew - 0.1, 0.04, ed - 0.1), "elevator_metal"
        )
        self.elev_ceiling = self.rigid_mgr.add_object_by_template_id(ceiling_tid)
        self.elev_ceiling.translation = _v3(ex, 2.3, ez)
        self.elev_ceiling.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # Elevator handrails
        for side in [-1, 1]:
            rail_tid = self._get_box_template(
                f"elev_rail_{side}", (0.04, 0.04, ed - 0.3), "railing"
            )
            rail = self.rigid_mgr.add_object_by_template_id(rail_tid)
            rail.translation = _v3(ex + side * (ew / 2 - 0.15), 0.9, ez)
            rail.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    def update_elevator(self, dt: float, target_floor: int = None):
        """Update elevator position. Call each physics step."""
        if self.elevator_platform is None:
            return

        if target_floor is not None:
            self.elevator_target_floor = target_floor

        target_y = self.elevator_target_floor * self.config.floor_height + 0.04
        cur_t = self.elevator_platform.translation
        current_y = cur_t[1]
        speed = 1.5  # m/s

        diff = target_y - current_y
        if abs(diff) < 0.01:
            self.elevator_platform.translation = _v3(cur_t[0], target_y, cur_t[2])
            return

        move = min(speed * dt, abs(diff)) * (1 if diff > 0 else -1)
        new_y = current_y + move

        self.elevator_platform.translation = _v3(cur_t[0], new_y, cur_t[2])
        if self.elev_ceiling is not None:
            ceil_t = self.elev_ceiling.translation
            self.elev_ceiling.translation = _v3(ceil_t[0], new_y + 2.26, ceil_t[2])

    # ------------------------------------------------------------------
    # Furniture
    # ------------------------------------------------------------------

    def _build_furniture(self):
        """Populate all rooms with realistic furniture."""
        for room in self.rooms:
            if room.room_type == "living":
                self._build_living_room(room)
            elif room.room_type == "kitchen":
                self._build_kitchen(room)
            elif room.room_type == "dining":
                self._build_dining(room)
            elif room.room_type == "bedroom":
                self._build_bedroom(room)
            elif room.room_type == "bathroom":
                self._build_bathroom(room)
            elif room.room_type == "foyer":
                self._build_foyer(room)
            elif room.room_type == "corridor":
                pass  # minimal furniture in corridors

    def _build_living_room(self, room: Room):
        """Build living room furniture: sofa, coffee table, TV, bookshelf, etc."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # L-shaped sofa against the back wall
        # Main sofa
        self._add_static_box(
            "sofa_main",
            (x0, y + 0.35, z0 - room.depth / 2 + 0.5),
            (3.0, 0.4, 0.9),
            "sofa",
        )
        # Sofa back
        self._add_static_box(
            "sofa_back",
            (x0, y + 0.7, z0 - room.depth / 2 + 0.15),
            (3.0, 0.6, 0.2),
            "sofa",
        )
        # Sofa cushions
        for i in range(3):
            self._add_static_box(
                f"sofa_cushion_{i}",
                (x0 - 0.9 + i * 0.9, y + 0.55, z0 - room.depth / 2 + 0.5),
                (0.7, 0.15, 0.7),
                "sofa_cushion",
            )

        # Side chaise
        self._add_static_box(
            "sofa_chaise",
            (x0 + 1.7, y + 0.35, z0 - room.depth / 2 + 1.3),
            (0.9, 0.4, 1.6),
            "sofa",
        )
        self._add_static_box(
            "sofa_chaise_back",
            (x0 + 2.05, y + 0.7, z0 - room.depth / 2 + 1.3),
            (0.2, 0.6, 1.6),
            "sofa",
        )

        # Coffee table
        self._add_static_box(
            "coffee_table_top",
            (x0, y + 0.42, z0 - 0.5),
            (1.4, 0.06, 0.7),
            "table_wood",
        )
        for dx in [-0.6, 0.6]:
            for dz in [-0.25, 0.25]:
                self._add_static_box(
                    f"coffee_leg_{dx}_{dz}",
                    (x0 + dx, y + 0.2, z0 - 0.5 + dz),
                    (0.06, 0.4, 0.06),
                    "table_wood",
                )

        # TV stand and TV against the front wall
        self._add_static_box(
            "tv_stand",
            (x0, y + 0.3, z0 + room.depth / 2 - 0.25),
            (2.2, 0.5, 0.45),
            "cabinet_wood",
        )
        self._add_static_box(
            "tv",
            (x0, y + 1.0, z0 + room.depth / 2 - 0.05),
            (1.6, 0.9, 0.06),
            "tv",
        )

        # Bookshelf on side wall
        self._add_static_box(
            "bookshelf",
            (x0 + room.width / 2 - 0.25, y + 1.1, z0 + 0.5),
            (0.4, 2.2, 1.2),
            "bookshelf",
        )
        # Shelf lines
        for i in range(4):
            self._add_static_box(
                f"shelf_{i}",
                (x0 + room.width / 2 - 0.25, y + 0.4 + i * 0.5, z0 + 0.5),
                (0.42, 0.02, 1.15),
                "bookshelf",
            )

        # Area rug
        self._add_static_box(
            "rug_living",
            (x0, y + 0.01, z0 - 0.5),
            (3.5, 0.02, 2.5),
            "rug",
        )

        # Floor lamp
        self._add_static_cylinder(
            "lamp_base", (x0 - 2.0, y + 0.02, z0 - room.depth / 2 + 0.5),
            0.15, 0.04, "lamp_base",
        )
        self._add_static_cylinder(
            "lamp_pole", (x0 - 2.0, y + 0.8, z0 - room.depth / 2 + 0.5),
            0.02, 1.5, "lamp_base",
        )
        self._add_static_box(
            "lamp_shade",
            (x0 - 2.0, y + 1.55, z0 - room.depth / 2 + 0.5),
            (0.35, 0.3, 0.35),
            "lamp_shade",
        )

        # Plant in corner
        self._add_static_cylinder(
            "plant_pot", (x0 - room.width / 2 + 0.5, y + 0.2, z0 - room.depth / 2 + 0.5),
            0.2, 0.4, "plant_pot",
        )
        self._add_static_box(
            "plant_leaves",
            (x0 - room.width / 2 + 0.5, y + 0.7, z0 - room.depth / 2 + 0.5),
            (0.6, 0.7, 0.6),
            "plant_green",
        )

    def _build_kitchen(self, room: Room):
        """Build kitchen: cabinets, counter, appliances, island."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # Base cabinets along back wall
        for i in range(5):
            cx = x0 - 3.0 + i * 1.0
            self._add_static_box(
                f"cabinet_base_{i}",
                (cx, y + 0.45, z0 - room.depth / 2 + 0.3),
                (0.9, 0.9, 0.6),
                "cabinet_wood",
            )

        # Countertop
        self._add_static_box(
            "countertop",
            (x0 - 1.0, y + 0.92, z0 - room.depth / 2 + 0.3),
            (5.0, 0.05, 0.65),
            "countertop",
        )

        # Upper cabinets
        for i in range(4):
            cx = x0 - 2.5 + i * 1.0
            self._add_static_box(
                f"cabinet_upper_{i}",
                (cx, y + 1.8, z0 - room.depth / 2 + 0.2),
                (0.9, 0.8, 0.4),
                "cabinet_wood",
            )

        # Refrigerator
        self._add_static_box(
            "fridge",
            (x0 + 3.2, y + 1.0, z0 - room.depth / 2 + 0.35),
            (0.8, 2.0, 0.7),
            "appliance",
        )

        # Oven/stove
        self._add_static_box(
            "stove",
            (x0 - 2.5, y + 0.47, z0 - room.depth / 2 + 0.3),
            (0.85, 0.85, 0.6),
            "appliance_dark",
        )

        # Sink
        self._add_static_box(
            "sink",
            (x0 - 0.5, y + 0.95, z0 - room.depth / 2 + 0.3),
            (0.7, 0.05, 0.5),
            "sink",
        )

        # Kitchen island
        self._add_static_box(
            "island_base",
            (x0 + 0.5, y + 0.45, z0 + 0.3),
            (2.0, 0.9, 0.8),
            "cabinet_wood",
        )
        self._add_static_box(
            "island_top",
            (x0 + 0.5, y + 0.92, z0 + 0.3),
            (2.1, 0.05, 0.9),
            "countertop",
        )

        # Bar stools
        for i in range(3):
            sx = x0 - 0.2 + i * 0.6
            self._add_static_box(
                f"stool_seat_{i}",
                (sx, y + 0.65, z0 + 1.0),
                (0.35, 0.05, 0.35),
                "chair_wood",
            )
            self._add_static_cylinder(
                f"stool_pole_{i}",
                (sx, y + 0.32, z0 + 1.0),
                0.03, 0.6, "chair_wood",
            )

    def _build_dining(self, room: Room):
        """Build dining area: table and chairs."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # Dining table
        self._add_static_box(
            "dining_table_top",
            (x0, y + 0.75, z0),
            (1.8, 0.06, 1.0),
            "table_wood",
        )
        for dx in [-0.8, 0.8]:
            for dz in [-0.4, 0.4]:
                self._add_static_box(
                    f"dining_leg_{dx}_{dz}",
                    (x0 + dx, y + 0.37, z0 + dz),
                    (0.08, 0.75, 0.08),
                    "table_wood",
                )

        # Chairs (4 around table)
        chair_positions = [
            (0, -0.8, 0), (0, 0.8, 0),
            (-1.1, 0, 90), (1.1, 0, -90),
        ]
        for i, (cx, cz, rot_y) in enumerate(chair_positions):
            # Seat
            self._add_static_box(
                f"chair_seat_{i}",
                (x0 + cx, y + 0.45, z0 + cz),
                (0.4, 0.05, 0.4),
                "chair_wood",
            )
            # Back
            back_offset = 0.2 if rot_y == 0 else -0.2 if abs(rot_y) > 0 else 0
            self._add_static_box(
                f"chair_back_{i}",
                (x0 + cx, y + 0.8, z0 + cz + back_offset),
                (0.4, 0.7, 0.05),
                "chair_wood",
            )
            # Legs
            for dx in [-0.15, 0.15]:
                for dz in [-0.15, 0.15]:
                    self._add_static_box(
                        f"chair_leg_{i}_{dx}_{dz}",
                        (x0 + cx + dx, y + 0.22, z0 + cz + dz),
                        (0.04, 0.45, 0.04),
                        "chair_wood",
                    )

    def _build_bedroom(self, room: Room):
        """Build bedroom: bed, nightstands, wardrobe, dresser."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # Bed (king size 1.8x2.0)
        bed_x = x0 - 1.0
        bed_z = z0 - room.depth / 2 + 1.2

        # Bed frame
        self._add_static_box(
            "bed_frame",
            (bed_x, y + 0.25, bed_z),
            (1.9, 0.35, 2.1),
            "bed_frame",
        )
        # Mattress
        self._add_static_box(
            "mattress",
            (bed_x, y + 0.5, bed_z),
            (1.8, 0.2, 2.0),
            "mattress",
        )
        # Headboard
        self._add_static_box(
            "headboard",
            (bed_x, y + 1.0, bed_z - 1.05),
            (1.9, 1.2, 0.1),
            "bed_frame",
        )
        # Pillows
        for i in [-0.45, 0.45]:
            self._add_static_box(
                f"pillow_{i}",
                (bed_x + i, y + 0.65, bed_z - 0.75),
                (0.6, 0.12, 0.4),
                "pillow",
            )
        # Blanket
        self._add_static_box(
            "blanket",
            (bed_x, y + 0.62, bed_z + 0.3),
            (1.75, 0.06, 1.4),
            "sofa",
        )

        # Nightstands
        for i, nx in enumerate([bed_x - 1.2, bed_x + 1.2]):
            self._add_static_box(
                f"nightstand_{i}",
                (nx, y + 0.3, bed_z - 0.8),
                (0.45, 0.6, 0.4),
                "cabinet_wood",
            )
            # Lamp on nightstand
            self._add_static_cylinder(
                f"bed_lamp_base_{i}",
                (nx, y + 0.65, bed_z - 0.8),
                0.08, 0.03, "lamp_base",
            )
            self._add_static_cylinder(
                f"bed_lamp_pole_{i}",
                (nx, y + 0.85, bed_z - 0.8),
                0.015, 0.35, "lamp_base",
            )
            self._add_static_box(
                f"bed_lamp_shade_{i}",
                (nx, y + 1.05, bed_z - 0.8),
                (0.2, 0.18, 0.2),
                "lamp_shade",
            )

        # Wardrobe
        self._add_static_box(
            "wardrobe",
            (x0 + room.width / 2 - 0.35, y + 1.1, z0 + 0.5),
            (0.6, 2.2, 2.0),
            "cabinet_wood",
        )

        # Dresser with mirror
        self._add_static_box(
            "dresser",
            (x0 + 1.5, y + 0.4, z0 + room.depth / 2 - 0.25),
            (1.4, 0.8, 0.5),
            "cabinet_wood",
        )
        self._add_static_box(
            "mirror",
            (x0 + 1.5, y + 1.5, z0 + room.depth / 2 - 0.02),
            (1.0, 1.2, 0.03),
            "window_glass",
        )

        # Rug under bed
        self._add_static_box(
            "rug_bedroom",
            (bed_x, y + 0.01, bed_z + 0.5),
            (2.5, 0.02, 2.0),
            "rug",
        )

    def _build_bathroom(self, room: Room):
        """Build bathroom: toilet, sink, bathtub/shower."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # Bathtub
        self._add_static_box(
            "bathtub",
            (x0 - 0.8, y + 0.3, z0 - room.depth / 2 + 0.4),
            (0.8, 0.6, 1.7),
            "bathtub",
        )
        # Tub interior (slightly different color)
        self._add_static_box(
            "tub_interior",
            (x0 - 0.8, y + 0.55, z0 - room.depth / 2 + 0.4),
            (0.65, 0.1, 1.55),
            "sink",
        )

        # Toilet
        self._add_static_box(
            "toilet_base",
            (x0 + 0.8, y + 0.2, z0 - room.depth / 2 + 0.3),
            (0.4, 0.4, 0.6),
            "toilet",
        )
        self._add_static_box(
            "toilet_tank",
            (x0 + 0.8, y + 0.55, z0 - room.depth / 2 + 0.05),
            (0.45, 0.45, 0.2),
            "toilet",
        )

        # Sink vanity
        self._add_static_box(
            "vanity",
            (x0 + 0.8, y + 0.4, z0 + 0.3),
            (0.7, 0.8, 0.5),
            "cabinet_wood",
        )
        self._add_static_box(
            "sink_basin",
            (x0 + 0.8, y + 0.82, z0 + 0.3),
            (0.6, 0.12, 0.4),
            "sink",
        )
        # Mirror
        self._add_static_box(
            "bath_mirror",
            (x0 + 0.8, y + 1.3, z0 + 0.55),
            (0.6, 0.8, 0.03),
            "window_glass",
        )

        # Shower glass partition
        self._add_static_box(
            "shower_glass",
            (x0 - 0.3, y + 1.0, z0 - room.depth / 2 + 0.4),
            (0.02, 2.0, 1.7),
            "window_glass",
        )

    def _build_foyer(self, room: Room):
        """Build foyer: console table, mirror, shoe cabinet."""
        x0, z0 = room.x, room.z
        y = room.floor * self.config.floor_height

        # Console table
        self._add_static_box(
            "console_top",
            (x0 - 0.5, y + 0.75, z0 + room.depth / 2 - 0.2),
            (1.2, 0.05, 0.35),
            "table_wood",
        )
        self._add_static_box(
            "console_base",
            (x0 - 0.5, y + 0.37, z0 + room.depth / 2 - 0.2),
            (1.1, 0.7, 0.3),
            "cabinet_wood",
        )

        # Mirror
        self._add_static_box(
            "foyer_mirror",
            (x0 - 0.5, y + 1.4, z0 + room.depth / 2 - 0.02),
            (0.8, 1.0, 0.03),
            "window_glass",
        )

        # Shoe cabinet near entrance
        self._add_static_box(
            "shoe_cabinet",
            (x0 + 0.8, y + 0.5, z0 - room.depth / 2 + 0.2),
            (0.8, 1.0, 0.35),
            "cabinet_wood",
        )

    def _build_exterior_details(self):
        """Add exterior details: door at entrance, light fixtures."""
        y = 0
        # Front door
        self._add_static_box(
            "front_door",
            (4.0, y + DOOR_HEIGHT / 2, -self.config.villa_depth / 2 + 0.05),
            (1.8, DOOR_HEIGHT, 0.08),
            "door_wood",
        )
        # Door frame
        self._add_static_box(
            "door_frame_t",
            (4.0, y + DOOR_HEIGHT + 0.05, -self.config.villa_depth / 2 + 0.05),
            (2.0, 0.1, 0.12),
            "frame_white",
        )

        # Exterior lights
        for dx in [-1.2, 1.2]:
            self._add_static_box(
                f"ext_light_{dx}",
                (4.0 + dx, y + 2.3, -self.config.villa_depth / 2 + 0.1),
                (0.15, 0.25, 0.1),
                "lamp_shade",
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_spawn_points(self) -> List[Tuple[float, float, float]]:
        """Return valid spawn points (one per floor near stairs/elevator)."""
        points = []
        for floor in range(self.config.num_floors):
            y = floor * self.config.floor_height + 0.02
            points.append((0.0, y, 0.0))  # center of living area
        return points

    def _get_stair_waypoints(self, from_floor: int):
        """Generate waypoints for climbing stairs from from_floor to from_floor+1."""
        y_base = from_floor * self.config.floor_height
        stair_x = -5.75
        flight2_x = stair_x + STAIR_WIDTH + 0.3
        wps = []

        # Approach stairs
        wps.append((-5.5, y_base, -2.0))
        wps.append((stair_x, y_base + 0.05, -4.3))

        # First flight
        for i in range(8):
            t = i / 7.0
            y = y_base + t * self.config.floor_height / 2
            z = -4.5 + t * 3.0
            wps.append((stair_x, y + 0.05, z))

        # Landing
        landing_y = y_base + self.config.floor_height / 2
        wps.append((stair_x, landing_y + 0.05, -1.0))
        wps.append((flight2_x, landing_y + 0.05, -0.7))

        # Second flight
        for i in range(8):
            t = i / 7.0
            y = landing_y + t * self.config.floor_height / 2
            z = -0.5 - t * 3.0
            wps.append((flight2_x, y + 0.05, z))

        # Exit stairs
        y_next = (from_floor + 1) * self.config.floor_height
        wps.append((flight2_x, y_next + 0.05, -3.8))
        wps.append((-3.0, y_next, -2.0))

        return wps

    def get_navigation_waypoints(self) -> List[Tuple[float, float, float]]:
        """Return a scripted path that explores the villa."""
        waypoints = []

        # Ground floor tour
        waypoints.extend([
            (0, 0, 0),
            (3, 0, -2),
            (3, 0, 2),
            (0, 0, 3),
            (-3, 0, 3),
            (-3, 0, 0),
        ])

        # Climb stairs and explore each floor
        for floor in range(self.config.num_floors - 1):
            y = floor * self.config.floor_height
            waypoints.extend(self._get_stair_waypoints(floor))

            # Explore the floor
            yf = (floor + 1) * self.config.floor_height
            waypoints.extend([
                (0, yf, 0),
                (3, yf, -2),
                (3, yf, 2),
                (0, yf, 3),
                (-3, yf, 2),
                (-3, yf, -2),
            ])

        return waypoints
