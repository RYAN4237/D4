import threading
import time

# Ensure repo root is importable
import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from final import demo


def make_task(name):
    return lambda: print(f"TASK EXECUTED: {name}")


def run_test():
    print("Starting keyboard_worker thread (daemon)")
    t = threading.Thread(target=demo.keyboard_worker, daemon=True)
    t.start()

    # Enqueue moves first (lowest priority)
    print("Enqueue: MOVE1, MOVE2")
    demo.move_queue.put(make_task('MOVE1'))
    demo.move_queue.put(make_task('MOVE2'))

    # Short pause then enqueue attacks (higher priority)
    time.sleep(0.1)
    print("Enqueue: ATTACK1, ATTACK2")
    demo.attack_queue.put(make_task('ATTACK1'))
    demo.attack_queue.put(make_task('ATTACK2'))

    # Short pause then enqueue a misc key
    time.sleep(0.1)
    print("Enqueue: KEY1")
    demo.key_queue.put(make_task('KEY1'))

    # After a short delay enqueue HP (highest priority)
    time.sleep(0.2)
    print("Enqueue: HP1 (should preempt others)")
    demo.hp_queue.put(make_task('HP1'))

    # Allow some time for processing
    time.sleep(1.0)

    # Send shutdown sentinels to allow worker to exit cleanly
    demo.hp_queue.put(None)
    demo.key_queue.put(None)
    demo.attack_queue.put(None)
    demo.move_queue.put(None)

    # Wait for queues to be drained
    demo.hp_queue.join()
    demo.key_queue.join()
    demo.attack_queue.join()
    demo.move_queue.join()

    print("Queues drained, test complete")


if __name__ == '__main__':
    run_test()

