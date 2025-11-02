import sys
sys.path.insert(0, r'C:\Repo\D4')
from final import minimap
from functools import partial
import queue

print('--- Test 1: queue without sentinel ---')
q = queue.Queue()
q.put(partial(print, 'old1'))
q.put(partial(print, 'old2'))
print('Before helper, queue size:', q.qsize())
minimap._enqueue_latest_move(q, partial(print, 'new'))
print('After helper, queue size:', q.qsize())
items = []
while not q.empty():
    it = q.get_nowait()
    items.append(it)
    q.task_done()
print('Items repr:')
for it in items:
    print(' ', repr(it))

print('\n--- Test 2: queue with sentinel in middle ---')
q2 = queue.Queue()
q2.put(partial(print, 'old1'))
q2.put(None)
q2.put(partial(print, 'old2'))
print('Before helper, queue size:', q2.qsize())
minimap._enqueue_latest_move(q2, partial(print, 'new_after_sentinel'))
print('After helper, queue size:', q2.qsize())
items2 = []
while not q2.empty():
    it = q2.get_nowait()
    items2.append(it)
    q2.task_done()
print('Items repr:')
for it in items2:
    print(' ', repr(it))

