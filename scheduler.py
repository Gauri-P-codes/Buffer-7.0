"""
scheduler.py
------------
Smart Assignment + Optimization Logic.
"""

import heapq
from queue_manager import QueueManager, ServiceCounter, Customer


class Scheduler:
    """
    Handles all smart decisions:
      1. Which counter should a new customer go to? (Greedy)
      2. Should we rebalance queues?                (Rebalancing)
      3. Are we in a peak-load period?              (Sliding Window)
      4. Should we switch to SJF mode?              (Adaptive Priority)
      5. Inject emergency VIP                       (Emergency Injection)
    """

    REBALANCE_THRESHOLD = 2
    SJF_THRESHOLD = 5
    PEAK_WINDOW = 5
    PEAK_SPIKE = 3

    def __init__(self, manager: QueueManager):
        self.manager = manager
        self.mode = "NORMAL"
        self.arrival_window = []
        self.is_peak = False

    def assign_customer(self, customer: Customer) -> int:
        """
        Greedy: Find the counter with the MINIMUM queue length.
        Ties broken by counter ID.
        """
        best_counter = min(
            self.manager.counters,
            key=lambda c: c.total_length()
        )
        best_counter.add_customer(customer)

        label = "⭐VIP" if customer.is_vip else "👤"
        self.manager.log(
            f"{label} {customer.name} → Counter-{best_counter.counter_id} "
            f"(queue len={best_counter.total_length()})"
        )

        self._record_arrival()
        self._check_adaptive_mode()

        return best_counter.counter_id

    def serve_next_global(self) -> tuple[Customer | None, int | None]:
        """
        Serve from the busiest counter in NORMAL mode.
        In SJF mode: serve the shortest-job customer across counters.
        """
        if self.mode == "SJF":
            return self._serve_sjf()
        return self._serve_normal()

    def _serve_normal(self) -> tuple[Customer | None, int | None]:
        """Normal mode: serve from the counter with the most people."""
        busiest = max(self.manager.counters, key=lambda c: c.total_length())

        if busiest.total_length() == 0:
            return None, None

        customer = busiest.serve_next()

        self.manager.log(
            f"✅ [FCFS] Served {customer} from Counter-{busiest.counter_id}"
        )

        self._check_adaptive_mode()
        return customer, busiest.counter_id

    def _serve_sjf(self) -> tuple[Customer | None, int | None]:
        """Shortest Job First across all counters."""
        candidates = []

        for counter in self.manager.counters:
            front = counter.peek_next()
            if front:
                heapq.heappush(candidates, (front.service_time, counter.counter_id, counter))

        if not candidates:
            return None, None

        _, _, chosen_counter = heapq.heappop(candidates)
        customer = chosen_counter.serve_next()

        self.manager.log(
            f"⚡ [SJF] Served {customer} from Counter-{chosen_counter.counter_id} "
            f"(svc={customer.service_time}s)"
        )

        self._check_adaptive_mode()
        return customer, chosen_counter.counter_id

    def rebalance(self) -> int:
        """
        Move customers from overloaded counters to underloaded ones.
        Only moves NORMAL customers.
        """
        moved = 0
        counters = self.manager.counters

        for _ in range(len(counters)):
            longest = max(counters, key=lambda c: c.total_length())
            shortest = min(counters, key=lambda c: c.total_length())

            diff = longest.total_length() - shortest.total_length()
            if diff <= self.REBALANCE_THRESHOLD:
                break

            if longest.queue:
                customer = longest.queue.pop()
                shortest.queue.appendleft(customer)
                moved += 1
                self.manager.log(
                    f"🔄 Rebalanced: {customer.name} "
                    f"Counter-{longest.counter_id}→Counter-{shortest.counter_id}"
                )
            else:
                break

        if moved == 0:
            self.manager.log("✅ Queues already balanced — no rebalancing needed.")
        else:
            self.manager.log(f"🔄 Rebalancing complete. Moved {moved} customer(s).")

        self._check_adaptive_mode()
        return moved

    def _record_arrival(self):
        """Maintain a sliding window of arrival timestamps and detect spikes."""
        import time

        now = time.time()
        self.arrival_window.append(now)

        cutoff = now - self.PEAK_WINDOW
        self.arrival_window = [t for t in self.arrival_window if t >= cutoff]

        if len(self.arrival_window) >= self.PEAK_SPIKE:
            if not self.is_peak:
                self.is_peak = True
                self.manager.log(
                    f"🚨 PEAK DETECTED — {len(self.arrival_window)} arrivals "
                    f"in last {self.PEAK_WINDOW}s!"
                )
        else:
            self.is_peak = False

    def _check_adaptive_mode(self):
        """Switch to SJF when queues are overloaded."""
        max_len = max(c.total_length() for c in self.manager.counters)
        if max_len >= self.SJF_THRESHOLD and self.mode != "SJF":
            self.mode = "SJF"
            self.manager.log(
                f"⚡ MODE SWITCH → SJF (queue length {max_len} ≥ threshold {self.SJF_THRESHOLD})"
            )
        elif max_len < self.SJF_THRESHOLD and self.mode != "NORMAL":
            self.mode = "NORMAL"
            self.manager.log("🔵 MODE SWITCH → NORMAL (load reduced)")

    def inject_emergency(self, name: str = "Emergency") -> int:
        """
        Instantly create and insert an ultra-high-priority customer.
        """
        customer = Customer(name, is_vip=True, service_time=1)
        customer.priority = -1
        customer.name = f"🚑 {name}"

        best = min(self.manager.counters, key=lambda c: c.total_length())
        heapq.heappush(best.priority_heap, (customer.priority, customer))

        self.manager.log(
            f"🚨 EMERGENCY INJECTION: {customer.name} → Counter-{best.counter_id}"
        )
        return best.counter_id

    def predict_service_time(self, counter: ServiceCounter) -> float:
        """Weighted average of past service times."""
        history = counter._service_history
        if not history:
            return counter.avg_service_time

        weights = list(range(1, len(history) + 1))
        weighted_sum = sum(w * t for w, t in zip(weights, history))
        total_weight = sum(weights)
        return round(weighted_sum / total_weight, 2)

    def status_summary(self) -> dict:
        """Return dashboard metrics."""
        counters = self.manager.counters
        return {
            "mode": self.mode,
            "is_peak": self.is_peak,
            "total_customers": self.manager.total_customers(),
            "queue_lengths": [c.total_length() for c in counters],
            "wait_times": [c.estimated_wait_time() for c in counters],
            "served_counts": [c.served_count for c in counters],
            "predicted_times": [self.predict_service_time(c) for c in counters],
        }
