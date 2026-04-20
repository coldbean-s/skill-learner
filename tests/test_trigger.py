from skill_learner.trigger import ReviewTrigger


class TestReviewTrigger:
    def test_initial_state_not_ready(self):
        t = ReviewTrigger(nudge_interval=10)
        assert not t.should_review()

    def test_iteration_threshold_triggers(self):
        t = ReviewTrigger(nudge_interval=5)
        for _ in range(5):
            t.tick_iteration()
        assert t.should_review()

    def test_turn_threshold_triggers(self):
        t = ReviewTrigger(nudge_interval=3)
        for _ in range(3):
            t.tick_turn()
        assert t.should_review()

    def test_mixed_counters_below_threshold(self):
        t = ReviewTrigger(nudge_interval=10)
        for _ in range(4):
            t.tick_iteration()
        for _ in range(4):
            t.tick_turn()
        assert not t.should_review()

    def test_reset_clears_both(self):
        t = ReviewTrigger(nudge_interval=3)
        for _ in range(3):
            t.tick_iteration()
        assert t.should_review()
        t.reset()
        assert not t.should_review()

    def test_zero_interval_never_triggers(self):
        t = ReviewTrigger(nudge_interval=0)
        for _ in range(100):
            t.tick_iteration()
        assert not t.should_review()
