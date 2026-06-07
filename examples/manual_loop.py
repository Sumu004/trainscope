"""Runnable demo with no ML deps — simulates a dataloader-bound training loop.

python examples/manual_loop.py
trainscope analyze runs/demo
"""

import time

from trainscope import Profiler


def fake_dataloader(n):
    for i in range(n):
        time.sleep(0.012)  # slow input pipeline (the bottleneck we want flagged)
        yield i


def main():
    prof = Profiler("runs/demo", name="demo", warmup=5)
    prof.start()
    for _ in prof.iter_data(fake_dataloader(100)):
        with prof.step():
            time.sleep(0.004)  # "forward"
            prof.mark("forward")
            time.sleep(0.006)  # "backward"
            prof.mark("backward")
            time.sleep(0.001)  # "optimizer"
            prof.mark("optimizer")
    prof.finish()
    print("Recorded run to runs/demo — now run:  trainscope analyze runs/demo")


if __name__ == "__main__":
    main()
