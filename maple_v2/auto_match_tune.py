import sys
import os
import cv2
import time
from queue import Queue

# ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from final import templates

# find image
candidates = ['img.png', 'map.png', 'map_grey.png']
img_path = None
for fn in candidates:
    p = os.path.join(repo_root, fn)
    if os.path.exists(p):
        img_path = p
        break
if img_path is None:
    print('No image found in repo root to test against (img.png/map.png).')
    sys.exit(1)

img = cv2.imdecode(__import__('numpy').fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
if img is None:
    print('Failed to load image')
    sys.exit(1)

templates_dir = os.path.join(repo_root, 'monsters')
if not os.path.exists(templates_dir):
    print('Templates dir not found:', templates_dir)
    sys.exit(1)

templates_cache = templates.load_templates_cache(templates_dir)
print('Loaded templates:', len(templates_cache))

# detectors to try
detectors = []
if hasattr(cv2, 'SIFT_create'):
    detectors.append(('SIFT', cv2.SIFT_create()))
if hasattr(cv2, 'ORB_create'):
    detectors.append(('ORB', cv2.ORB_create(nfeatures=1000)))
if not detectors:
    print('No detectors available')
    sys.exit(1)

match_ratios = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

# center pos as before
h, w = img.shape[:2]
ch_pos = (w // 2, h // 2)
key_queue = Queue()

out_results = []

for name, det in detectors:
    for ratio in match_ratios:
        print(f'Testing detector={name}, match_ratio={ratio} (debug images will be saved)')
        start = time.time()
        goods = templates.template_match_monster(img.copy(), det, templates_cache, ch_pos, is_left=False, match_ratio=ratio, key_queue=key_queue, debug=True)
        elapsed = time.time() - start
        print(f'  -> goods: {len(goods)}, elapsed={elapsed:.2f}s')
        out_results.append((name, ratio, len(goods), elapsed))

print('\nSummary:')
for r in out_results:
    print(r)

print('\nCheck final/ for debug_crop_* and debug_kp_* files for the last runs.')

