import sys
import os
# ensure repo root is on sys.path so `import final` works when running this script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import cv2
from queue import Queue
from final import templates, capture

# find an image to use
candidates = ['img.png', 'map.png', 'map_grey.png']
root = repo_root
img_path = None
for fn in candidates:
    p = os.path.join(root, fn)
    if os.path.exists(p):
        img_path = p
        break

if img_path is None:
    print('No candidate image found in repo root. Please place an image named img.png or map.png next to repo root.')
    exit(1)

# load image
img = cv2.imdecode(__import__('numpy').fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
if img is None:
    print('Failed to load image:', img_path)
    exit(1)

# load templates cache
templates_dir = os.path.join(root, 'monsters')
if not os.path.exists(templates_dir):
    print('Templates dir not found:', templates_dir)
    exit(1)

templates_cache = templates.load_templates_cache(templates_dir)
print('Loaded templates:', len(templates_cache))

# create detector
if hasattr(cv2, 'SIFT_create'):
    detector = cv2.SIFT_create()
    print('Using SIFT')
elif hasattr(cv2, 'ORB_create'):
    detector = cv2.ORB_create(nfeatures=1000)
    print('Using ORB')
else:
    print('No feature detector available')
    exit(1)

# choose a character pos roughly center
h, w = img.shape[:2]
ch_pos = (w // 2, h // 2)

# call function synchronously
key_queue = Queue()
print('Calling template_match_monster...')
res = templates.template_match_monster(img.copy(), detector, templates_cache, ch_pos, is_left=False, key_queue=key_queue)
print('Result goods:', res)

# show image with drawn results (if any)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
