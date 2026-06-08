"""The headline demo — a finding no single-axis tool can produce.

Simulates a training run that hits an optimization blow-up around step 70:
loss spikes, grad-norm explodes, AND the step time hitches — all at once.
pytscope correlates them into one finding. No ML deps required.

    python examples/cross_signal.py
    pytscope analyze runs/cross
"""

import math
import random
import time

from pytscope import Profiler

random.seed(0)


def main():
    prof = Profiler("runs/cross", name="cross-signal-demo", warmup=0)
    prof.start()

    for step in range(120):
        # Healthy training: loss decays, grad-norm steady — except a blow-up.
        loss = 2.0 * math.exp(-0.025 * step) + 0.05 + random.uniform(-0.01, 0.01)
        grad_norm = 1.0 + random.uniform(-0.1, 0.1)
        hitch = 0.0
        if 70 <= step <= 72:  # the correlated event
            loss *= 9.0
            grad_norm = 45.0
            hitch = 0.012  # a real step-time stall too

        for _batch in prof.iter_data([step]):  # trivial 1-item loader
            with prof.step():
                time.sleep(0.002 + hitch)  # "forward"
                prof.mark("forward")
                time.sleep(0.003)  # "backward"
                prof.mark("backward")
                prof.mark("optimizer")
                prof.log(loss=loss, grad_norm=grad_norm)

    prof.finish()
    print("Recorded runs/cross — now run:  pytscope analyze runs/cross")


if __name__ == "__main__":
    main()
