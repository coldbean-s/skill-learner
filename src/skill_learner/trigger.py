from __future__ import annotations


class ReviewTrigger:
    def __init__(self, nudge_interval: int = 10):
        self._iters = 0
        self._turns = 0
        self._nudge_interval = nudge_interval

    def tick_iteration(self) -> None:
        self._iters += 1

    def tick_turn(self) -> None:
        self._turns += 1

    def should_review(self) -> bool:
        if self._nudge_interval <= 0:
            return False
        return (self._iters >= self._nudge_interval
                or self._turns >= self._nudge_interval)

    def reset(self) -> None:
        self._iters = 0
        self._turns = 0
