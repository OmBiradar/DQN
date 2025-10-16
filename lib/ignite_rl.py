# lib/ignite_rl.py
from collections import deque
import time
import numpy as np

from ignite.engine import Engine, Events
from ignite.engine.events import EventEnum

class EpisodeEvents(EventEnum):
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"

# Map custom events to a state attribute so loggers/filters can recognize them
_EPISODE_EVENT_TO_ATTR = {
    EpisodeEvents.EPISODE_STARTED: "episode",
    EpisodeEvents.EPISODE_COMPLETED: "episode",
    EpisodeEvents.BOUND_REWARD_REACHED: "episode",
}

class PeriodEvents(EventEnum):
    ITERS_100_COMPLETED = "iters_100_completed"

_PERIOD_EVENT_TO_ATTR = {
    PeriodEvents.ITERS_100_COMPLETED: "iters_100_completed",
}

class EndOfEpisodeHandler:
    def __init__(self, exp_source, bound_avg_reward=None, avg_reward_len: int = 100):
        self.exp_source = exp_source
        self.bound_avg_reward = bound_avg_reward
        self.rewards = deque(maxlen=avg_reward_len)

    def attach(self, engine: Engine):
        # Register with event_to_attr so loggers accept these events
        engine.register_events(*EpisodeEvents, event_to_attr=_EPISODE_EVENT_TO_ATTR)

        @engine.on(Events.STARTED)
        def _init_state(_):
            st = engine.state
            st.episode = 0
            st.episode_reward = 0.0
            st.episode_steps = 0
            st.solved = False
            if not hasattr(st, "metrics") or st.metrics is None:
                st.metrics = {}
            st.metrics.setdefault("avg_reward", 0.0)
            st.metrics.setdefault("reward", 0.0)
            st.metrics.setdefault("steps", 0)

        @engine.on(Events.ITERATION_COMPLETED)
        def _check_episode_end(_):
            for reward, steps in self.exp_source.pop_rewards_steps():
                engine.state.episode += 1
                engine.state.episode_reward = float(reward)
                engine.state.episode_steps = int(steps)
                self.rewards.append(float(reward))

                avg = float(np.mean(self.rewards)) if len(self.rewards) > 0 else 0.0
                engine.state.metrics["reward"] = float(reward)
                engine.state.metrics["steps"] = int(steps)
                engine.state.metrics["avg_reward"] = avg

                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)

                if (
                    self.bound_avg_reward is not None
                    and len(self.rewards) == self.rewards.maxlen
                    and avg >= float(self.bound_avg_reward)
                ):
                    engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)

class EpisodeFPSHandler:
    def __init__(self):
        self._t0 = None

    def attach(self, engine: Engine):
        @engine.on(Events.STARTED)
        def _start(_):
            self._t0 = time.time()
            engine.state.metrics.setdefault("avg_fps", 0.0)
            engine.state.metrics.setdefault("time_passed", 0.0)

        @engine.on(Events.ITERATION_COMPLETED)
        def _update(_):
            elapsed = max(1e-6, time.time() - self._t0)
            iters = max(1, engine.state.iteration)
            engine.state.metrics["avg_fps"] = iters / elapsed
            engine.state.metrics["time_passed"] = elapsed

class PeriodicEvents:
    def attach(self, engine: Engine):
        # Register with event_to_attr; also initialize and increment the counter
        engine.register_events(*PeriodEvents, event_to_attr=_PERIOD_EVENT_TO_ATTR)

        @engine.on(Events.STARTED)
        def _init(_):
            engine.state.iters_100_completed = 0

        @engine.on(Events.ITERATION_COMPLETED)
        def _every_100(_):
            if engine.state.iteration % 100 == 0:
                engine.state.iters_100_completed += 1
                engine.fire_event(PeriodEvents.ITERS_100_COMPLETED)
