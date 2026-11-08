"""Record timestamp differences from pairs actually selected by a synchronizer."""
from collections import defaultdict, deque
import json
import time

import numpy as np


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


class SyncDiagnostics:
    """Periodically summarize matched-pair skew separately from arrival age."""

    def __init__(self, node):
        self.node = node
        self.samples = defaultdict(lambda: deque(maxlen=2000))
        self.last_log = time.monotonic()

    def clear(self):
        self.samples.clear()

    def record(self, observation, odometry, stream):
        observation_ns, odometry_ns = stamp_ns(observation), stamp_ns(odometry)
        delta_s = (odometry_ns - observation_ns) / 1e9
        age_s = (self.node.get_clock().now().nanoseconds - observation_ns) / 1e9
        self.samples[stream].append((abs(delta_s), age_s))
        if time.monotonic() - self.last_log >= 5.:
            self.log_summary()

    def summary(self):
        return {stream: dict(matches=len(samples),
                            abs_delta_ms_p50_p95_max=(np.percentile(
                                np.array(samples)[:, 0], [50, 95, 100]) * 1000).tolist(),
                            arrival_age_ms_p50_p95_max=(np.percentile(
                                np.array(samples)[:, 1], [50, 95, 100]) * 1000).tolist())
                for stream, samples in self.samples.items() if samples}

    def log_summary(self):
        summary = self.summary()
        if summary:
            self.node.get_logger().info('Accepted timestamp matches: ' + json.dumps(summary))
        self.last_log = time.monotonic()
