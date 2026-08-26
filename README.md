# habitat_scene_builder

Procedural multi-story villa scene generator for Habitat-Sim 0.3.1.

## Features

- Random 2-4 story villa layout with realistic room partitions (living room, kitchen, dining, bedrooms, bathrooms, foyer)
- U-shaped staircase connecting all floors with railings
- Functional elevator (kinematic platform, can be scripted)
- Bullet physics collision on all objects (walls, floor, stairs, furniture)
- Furniture: sofas, beds, tables, chairs, cabinets, appliances, bathroom fixtures, lamps, plants
- First-person and third-person camera views
- Video export via imageio-ffmpeg

## Requirements

- Habitat-Sim 0.3.1 built with `--with-bullet --headless`
- Python 3.9, magnum, numpy, imageio, imageio-ffmpeg

## Usage

```bash
python run_villa.py [--num_floors {2,3,4}] [--seed SEED] [--width W] [--height H] [--fps FPS]
```

If `--num_floors` is omitted, a random value between 2 and 4 is selected.

Output videos are saved to `output/`.

## Code Structure

- `villa_generator.py` — `VillaBuilder` class that constructs the scene from primitive collision shapes (cube, cylinder) using Habitat-Sim's Bullet physics backend
- `run_villa.py` — Simulator setup, agent navigation with collision avoidance, elevator animation, and video recording

## Architecture Notes

All geometry uses Habitat-Sim built-in primitive templates (`cubeSolid`, `cylinderSolid`) which have `use_mesh_collision=false`, enabling Bullet primitive colliders (box, cylinder). This ensures zero clipping during robot navigation. The agent uses a cylinder collision body (radius 0.3m, height 1.2m).
