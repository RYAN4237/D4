# D4

Game automation toolkit with two main tracks:
- `maple_v2/`: MapleStory minimap stitching, monster matching, training assets, and YOLO-based detection utilities.
- `poe2/`: Path of Exile 2 minimap capture, stitching, and A* route following with combat helpers.

## Quick start
1) Install Python 3.10+ and ffmpeg (for OpenCV video support if needed).
2) Install deps:
```powershell
cd C:\Users\Administrator\PycharmProjects\D4
python -m venv .venv
. .venv\Scripts\activate
pip install -r maple_v2\requirements.txt
```
3) (Optional) If using POE2 driver, place `driver.dll` at `poe2/driver.dll` (already referenced in code).

## Run examples
- MapleStory demo (stitch and detection preview):
```powershell
python -m maple_v2.demo
```
- Path of Exile 2 route follower (captures window titled "Path of Exile 2"):
```powershell
python -m poe2.route
```

## Controls & notes
- F1 in `poe2.route` sets a stop flag for the main loop.
- Window titles must match the target game (`CaptureScreen.get_hwnd`).
- Ensure in-game minimap is visible; capture regions are hard-coded (see `poe2/route.py`).

## Key components
- `maple_v2/minimap.py`, `stitch.py`, `minimap_calibrate.py`: minimap capture/stitching pipeline.
- `maple_v2/templates.py`, `monsters/`, `maps/`, `tag/`: matching data and labels for YOLO training.
- `maple_v2/yolo_train.py`: training entry for custom YOLO models.
- `maple_v2/test_*.py`: enqueue/priority helpers and monster matching tests.
- `poe2/stitch_new.py`: SmartMinimapStitcher worker used by `poe2.route`.
- `poe2/a_star.py`: pathfinding utilities (`a_star_new`, `mini_map_matching`).
- `poe2/attack.py`: simple combat loop (`attack_nearby_enemies`).
- `poe2/route.py`: orchestrates capture → stitch → locate → pathfind → move → attack.

## Troubleshooting
- No window found: verify the game is running and the title matches exactly.
- Blank/slow stitch: check capture bounds and monitor scaling (try 100% DPI); ensure minimap is not occluded.
- Pathfinding stuck: confirm `poe2/route.py` capture coordinates match your resolution; adjust offsets in code if needed.
- OpenCV windows not visible: they may be behind the game window; press `q` in the OpenCV window to exit.

## Roadmap / TODO
- Parameterize capture rectangles and hotkeys via config.
- Add automated tests around stitching and pathfinding.
- Document GPU/CPU requirements and add CI smoke tests.
