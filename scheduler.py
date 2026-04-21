"""
scheduler.py
------------
Smart Assignment + Optimization Logic.
"""

import heapq
import time
from queue_manager import QueueManager, ServiceCounter, Customer


class Scheduler:
    REBALANCE_THRESHOLD = 2
    PEAK_WINDOW = 5
    PEAK_SPIKE = 3

    def __init__(self, manager: QueueManager):
        self.manager = manager
        self.mode = "NORMAL"          # always NORMAL now, kept for UI compatibility
        self.arrival_window = []
        self.is_peak = False

    def assign_customer(self, customer: Customer) -> int:
        """Greedy: assign to counter with minimum queue length."""
        best_counter = min(self.manager.counters, key=lambda c: c.total_length())
        best_counter.add_customer(customer)
        label = "⭐VIP" if customer.is_vip else "👤"
        self.manager.log(
            f"{label} {customer.name} → Counter-{best_counter.counter_id} "
            f"(queue len={best_counter.total_length()})"
        )
        self._record_arrival()
        return best_counter.counter_id

    def serve_next_global(self) -> tuple[Customer | None, int | None]:
        """
        Single authoritative serve function.
        Priority chain (across ALL counters, irrespective of counter):
          Emergency (priority=-1) → VIP (priority=0) → Normal FCFS (priority=1)
        Normal customers always served by arrival_time (FCFS), never SJF.
        """
        priority_candidates = []   # Emergency + VIP
        normal_candidates   = []   # Normal customers

        for counter in self.manager.counters:
            front = counter.peek_next()
            if front is None:
                continue
            priority     = getattr(front, 'priority', 1)
            arrival_time = getattr(front, 'arrival_time', time.time())

            if priority < 1:
                heapq.heappush(priority_candidates,
                               (priority, arrival_time, counter.counter_id, counter))
            else:
                # FCFS — sort by arrival_time only
                heapq.heappush(normal_candidates,
                               (arrival_time, counter.counter_id, counter))

        if priority_candidates:
            priority, _, counter_id, chosen = heapq.heappop(priority_candidates)
            customer = chosen.serve_next()
            label = "🚨 EMERGENCY" if priority == -1 else "⭐ VIP"
            self.manager.log(
                f"✅ {label} {customer.name} served from Counter-{counter_id}"
            )
            return customer, counter_id

        if normal_candidates:
            _, counter_id, chosen = heapq.heappop(normal_candidates)
            customer = chosen.serve_next()
            self.manager.log(
                f"✅ [FCFS] 👤 {customer.name} served from Counter-{counter_id}"
            )
            return customer, counter_id

        return None, None

    # Compatibility aliases — UI calls these in some paths
    def _serve_normal(self):
        return self.serve_next_global()

    def serve_global_priority(self):
        return self.serve_next_global()

    def rebalance(self) -> int:
        """Move normal customers from overloaded to underloaded counters."""
        moved = 0
        counters = self.manager.counters

        for _ in range(len(counters)):
            longest  = max(counters, key=lambda c: c.total_length())
            shortest = min(counters, key=lambda c: c.total_length())

            if longest.total_length() - shortest.total_length() <= self.REBALANCE_THRESHOLD:
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
        return moved

    def _record_arrival(self):
        """Sliding window peak detection."""
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

    def inject_emergency(self, name: str = "Emergency") -> int:
        """Instantly insert an ultra-high-priority customer."""
        customer = Customer(name, is_vip=True, service_time=1)
        customer.priority = -1
        customer.arrival_time = time.time()
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
        return round(weighted_sum / sum(weights), 2)

    def status_summary(self) -> dict:
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
