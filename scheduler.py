"""
scheduler.py
------------
Smart Assignment + Optimization Logic.

DSA / Algorithm Concepts Used:
  - Greedy Algorithm       → assign to shortest queue
  - Sliding Window         → detect arrival spikes (peak detection)
  - Shortest Job First     → serve customer with least service time
  - Rebalancing            → move customers from long → short queues
"""

import heapq
import random
from queue_manager import QueueManager, ServiceCounter, Customer


# ─────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────
class Scheduler:
    """
    Handles all smart decisions:
      1. Which counter should a new customer go to? (Greedy)
      2. Should we rebalance queues?                (Rebalancing)
      3. Are we in a peak-load period?              (Sliding Window)
      4. Should we switch to SJF mode?              (Adaptive Priority)
      5. Inject emergency VIP                       (Emergency Injection)
    """

    # Thresholds (tweak these to change system behaviour)
    REBALANCE_THRESHOLD = 2   # max allowed difference in queue lengths before rebalancing
    SJF_THRESHOLD       = 5   # if any queue exceeds this length, switch to SJF mode
    PEAK_WINDOW         = 5   # sliding window size for peak detection
    PEAK_SPIKE          = 3   # if arrivals in window exceed this, it's a peak

    def __init__(self, manager: QueueManager):
        self.manager         = manager
        self.mode            = "NORMAL"          # "NORMAL" or "SJF"
        self.arrival_window  = []                # sliding window of recent arrivals
        self.is_peak         = False

    # ─────────────────────────────────────────
    # 1. GREEDY ASSIGNMENT — O(n) scan
    # ─────────────────────────────────────────
    def assign_customer(self, customer: Customer) -> int:
        """
        Greedy: Find the counter with the MINIMUM queue length.
        Ties broken by counter ID (first one wins).

        Returns the counter_id where the customer was placed.
        """
        # Find counter with minimum total_length — linear scan O(n counters)
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

        # Track arrival for peak detection
        self._record_arrival()

        # After adding, check if we need to switch mode
        self._check_adaptive_mode()

        return best_counter.counter_id

    # ─────────────────────────────────────────
    # 2. SERVE NEXT — respects current mode
    # ─────────────────────────────────────────
    def serve_next_global(self) -> tuple[Customer | None, int | None]:
        """
        Serve from the busiest counter (longest queue).
        In SJF mode: across ALL counters, serve the shortest job first.

        Returns (customer_served, counter_id) or (None, None) if empty.
        """
        if self.mode == "SJF":
            return self._serve_sjf()
        else:
            return self._serve_normal()

    def _serve_normal(self):
        """Normal mode: serve from the counter with the most people."""
        busiest = max(self.manager.counters, key=lambda c: c.total_length())
        if busiest.total_length() == 0:
            return None, None
        customer = busiest.serve_next()
        self.manager.log(f"✅ Served {customer} from Counter-{busiest.counter_id}")
        return customer, busiest.counter_id

    def _serve_sjf(self):
        """Shortest Job First: collect the front customer from every non-empty
        counter, pick the one with the smallest service_time, serve them.
        """
        candidates = []
        for counter in self.manager.counters:
            front = counter.peek_next()
            if front:
                # Include counter_id to break ties without comparing ServiceCounter objects
                heapq.heappush(candidates, (front.service_time, counter.counter_id, counter, front))

        if not candidates:
            return None, None

        service_time, counter_id, chosen_counter, customer = heapq.heappop(candidates)
        # Actually serve the customer (the one we peeked at)
        customer = chosen_counter.serve_next()
        self.manager.log(
            f"⚡[SJF] Served {customer} from Counter-{chosen_counter.counter_id} "
            f"(svc={customer.service_time}s)"
        )
        return customer, chosen_counter.counter_id

    # ─────────────────────────────────────────
    # 3. QUEUE REBALANCING — O(n) scan
    # ─────────────────────────────────────────
    def rebalance(self) -> int:
        """
        Move customers from overloaded counters to underloaded ones.
        Only moves NORMAL customers (VIP stays put — too important to shuffle).

        Returns number of customers moved.
        """
        moved = 0
        counters = self.manager.counters

        for _ in range(len(counters)):          # at most one pass per counter
            longest  = max(counters, key=lambda c: c.total_length())
            shortest = min(counters, key=lambda c: c.total_length())

            diff = longest.total_length() - shortest.total_length()
            if diff <= self.REBALANCE_THRESHOLD:
                break                           # already balanced enough

            # Move one normal customer from longest → shortest
            if longest.queue:                   # only normal queue customers
                customer = longest.queue.pop()  # take from BACK (least-wait impact)
                shortest.queue.appendleft(customer)  # put at FRONT of short queue
                moved += 1
                self.manager.log(
                    f"🔄 Rebalanced: {customer.name} "
                    f"Counter-{longest.counter_id}→Counter-{shortest.counter_id}"
                )
            else:
                break   # nothing to move (only VIPs remain in this counter)

        if moved == 0:
            self.manager.log("✅ Queues already balanced — no rebalancing needed.")
        else:
            self.manager.log(f"🔄 Rebalancing complete. Moved {moved} customer(s).")

        return moved

    # ─────────────────────────────────────────
    # 4. PEAK DETECTION — Sliding Window
    # ─────────────────────────────────────────
    def _record_arrival(self):
        """Maintain a sliding window of arrival counts and detect spikes."""
        import time
        now = time.time()
        self.arrival_window.append(now)

        # Remove entries outside the window
        cutoff = now - self.PEAK_WINDOW
        self.arrival_window = [t for t in self.arrival_window if t >= cutoff]

        # If arrivals within window exceed threshold → peak
        if len(self.arrival_window) >= self.PEAK_SPIKE:
            if not self.is_peak:
                self.is_peak = True
                self.manager.log(
                    f"🚨 PEAK DETECTED — {len(self.arrival_window)} arrivals "
                    f"in last {self.PEAK_WINDOW}s!"
                )
        else:
            self.is_peak = False

    # ─────────────────────────────────────────
    # 5. ADAPTIVE MODE — switch to SJF on overload
    # ─────────────────────────────────────────
    def _check_adaptive_mode(self):
        """If any queue exceeds SJF_THRESHOLD, switch to Shortest-Job-First."""
        max_len = max(c.total_length() for c in self.manager.counters)
        if max_len >= self.SJF_THRESHOLD and self.mode != "SJF":
            self.mode = "SJF"
            self.manager.log(
                f"⚡ MODE SWITCH → SJF (queue length {max_len} ≥ threshold {self.SJF_THRESHOLD})"
            )
        elif max_len < self.SJF_THRESHOLD and self.mode != "NORMAL":
            self.mode = "NORMAL"
            self.manager.log("🔵 MODE SWITCH → NORMAL (load reduced)")

    # ─────────────────────────────────────────
    # 6. EMERGENCY INJECTION
    # ─────────────────────────────────────────
    def inject_emergency(self, name: str = "Emergency") -> int:
        """
        Instantly create and insert an ultra-high-priority customer.
        Uses a special priority=−1 so they jump ahead of all VIPs.
        Returns the counter_id they were assigned to.
        """
        customer              = Customer(name, is_vip=True, service_time=1)
        customer.priority     = -1              # beats all other priorities
        customer.name         = f"🚑 {name}"

        # Always inject into the counter with the shortest queue
        best = min(self.manager.counters, key=lambda c: c.total_length())
        heapq.heappush(best.priority_heap, (customer.priority, customer))

        self.manager.log(
            f"🚨 EMERGENCY INJECTION: {customer.name} → Counter-{best.counter_id}"
        )
        return best.counter_id

    # ─────────────────────────────────────────
    # 7. SERVICE TIME PREDICTION
    # ─────────────────────────────────────────
    def predict_service_time(self, counter: ServiceCounter) -> float:
        """
        Simple prediction: weighted average of past service times.
        Recent times weighted more heavily (recency bias).

        Formula:  Σ(weight_i × time_i) / Σ(weight_i)
        """
        history = counter._service_history
        if not history:
            return counter.avg_service_time

        # weights: 1, 2, 3 … (more weight to recent entries)
        weights = list(range(1, len(history) + 1))
        weighted_sum = sum(w * t for w, t in zip(weights, history))
        total_weight = sum(weights)
        return round(weighted_sum / total_weight, 2)

    # ─────────────────────────────────────────
    # Status snapshot (used by UI)
    # ─────────────────────────────────────────
    def status_summary(self) -> dict:
        """Return a dict of key metrics for the dashboard."""
        counters = self.manager.counters
        return {
            "mode"            : self.mode,
            "is_peak"         : self.is_peak,
            "total_customers" : self.manager.total_customers(),
            "queue_lengths"   : [c.total_length() for c in counters],
            "wait_times"      : [c.estimated_wait_time() for c in counters],
            "served_counts"   : [c.served_count for c in counters],
            "predicted_times" : [self.predict_service_time(c) for c in counters],
        }
