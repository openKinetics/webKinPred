"""Host-local collection independent of long-running model supervision."""

import argparse
import json
import random
import time
from pathlib import Path

from tools.executor.agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    agent = Agent(json.loads(args.config.read_text()))
    while True:
        collected = False
        try:
            collected = agent.collect_artifacts()
        except Exception as error:
            # Errors are retryable; never print credentials or a request manifest.
            print(
                json.dumps(
                    {"event": "execution.cleanup_retry", "error_type": type(error).__name__}
                ),
                flush=True,
            )
        if args.once:
            return
        # Drain confirmed garbage promptly; jitter idle checks so optional work
        # does not continually collide with a supervisor's periodic host lock.
        time.sleep(5 if collected else random.uniform(25, 35))


if __name__ == "__main__":
    main()
