import argparse
from queue import Empty


def parse_args():
    p = argparse.ArgumentParser(description="Maple demo runner")
    p.add_argument("--window-title", default="MapleStory Worlds-Old School Maple")
    p.add_argument("--templates-dir", default=r"C:\Repo\D4\monsters")
    p.add_argument("--template-name", default=r"C:\Repo\D4\name.png")
    p.add_argument("--show-windows", action="store_true", default=True)
    p.add_argument("--hp-threshold", type=int, default=50)
    p.add_argument("--mp-threshold", type=int, default=50)
    p.add_argument("--buffer_time", type=int, default=60)
    p.add_argument("--debug", action="store_true", help="Enable debug mode (save debug crops/keypoints)", default=True)
    p.add_argument("--match-ratio", type=float, default=0.45, help="Lowe ratio for feature matching (0.5-0.95)")
    p.add_argument("--min-char-confidence", type=float, default=0.3, help="Minimum character template confidence to run monster detection")
    p.add_argument("--route-path", default=r"C:\Repo\D4\final\maps\zipangu")
    p.add_argument("--enable-yolo", action="store_true", default=True)
    return p.parse_args()


def _enqueue_latest_move(move_queue, p):
    """Drain older move tasks from move_queue (but preserve shutdown sentinel None) then enqueue p.
    This ensures the newest movement intent replaces any queued older movement commands.
    """
    if move_queue is None:
        return
    try:
        while True:
            try:
                old = move_queue.get_nowait()
            except Empty:
                break
            # if it's the sentinel, re-put it and stop draining
            if old is None:
                try:
                    move_queue.put(None)
                except Exception:
                    pass
                move_queue.task_done()
                break
            # otherwise discard the old movement task
            try:
                move_queue.task_done()
            except Exception:
                pass
    except Exception:
        # best effort only; if anything goes wrong, continue to enqueue
        pass
    try:
        move_queue.put(p)
    except Exception as e:
        print("minimap enqueue failed:", e)