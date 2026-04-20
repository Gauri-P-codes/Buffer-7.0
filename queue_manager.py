"""
queue_manager.py
----------------
Core Data Structures for the Queue System.

DSA Concepts Used:
  - collections.deque  → O(1) append/popleft for normal queues
  - heapq              → O(log n) push/pop for priority queue
  - list               → storing counter metadata
"""

import heapq
from collections import deque
import time


# ─────────────────────────────────────────────
# Customer class — one object per person
# ─────────────────────────────────────────────
class Customer:
    """Represents a single customer in the system."""

    _id_counter = 0  # class-level counter to give each customer a unique ID

    def __init__(self, name: str, is_vip: bool = False, service_time: int = 3):
        Customer._id_counter += 1
        self.customer_id   = Customer._id_counter
        self.name          = name
        self.is_vip        = is_vip
        self.service_time  = service_time   # how many seconds/ticks to serve
        self.arrival_time  = time.time()    # wall-clock arrival
        self.priority      = 0 if is_vip else 1  # lower number = higher priority

    def __repr__(self):
        tag = "⭐VIP" if self.is_vip else "👤Norm"
        return f"[{tag} #{self.customer_id} '{self.name}' svc={self.service_time}s]"

    # heapq compares tuples; we implement __lt__ so Customer objects are sortable
    def __lt__(self, other):
        return self.priority < other.priority


# ─────────────────────────────────────────────
# ServiceCounter — one counter / window
# ─────────────────────────────────────────────
class ServiceCounter:
    """
    Each ServiceCounter holds:
      - A deque (normal queue)          → FIFO, O(1) both ends
      - A heap  (priority queue)        → VIP-first, O(log n) ops
      - Metadata: busy flag, served count, average service time
    """

    def __init__(self, counter_id: int, avg_service_time: int = 3):
        self.counter_id        = counter_id
        self.queue: deque      = deque()         # normal customers
        self.priority_heap     = []              # heapq — VIP customers
        self.is_busy           = False
        self.served_count      = 0
        self.avg_service_time  = avg_service_time  # rolling average
        self._service_history  = []              # track past service times (for prediction)

    # ── Enqueue ──────────────────────────────
    def add_customer(self, customer: Customer):
        """Add customer to the right queue based on VIP status."""
        if customer.is_vip:
            # heapq uses a (priority, customer) tuple so ties break deterministically
            heapq.heappush(self.priority_heap, (customer.priority, customer))
        else:
            self.queue.append(customer)

    # ── Dequeue ──────────────────────────────
    def serve_next(self) -> Customer | None:
        """
        Serve the next customer.
        Priority order: VIP heap first → then normal deque.
        Returns the served Customer or None if empty.
        """
        if self.priority_heap:
            _, customer = heapq.heappop(self.priority_heap)
        elif self.queue:
            customer = self.queue.popleft()   # O(1) with deque
        else:
            return None

        # Update rolling average service time
        self._service_history.append(customer.service_time)
        if len(self._service_history) > 20:          # keep last 20 only
            self._service_history.pop(0)
        self.avg_service_time = sum(self._service_history) / len(self._service_history)

        self.served_count += 1
        self.is_busy = False
        return customer

    # ── Metrics ──────────────────────────────
    def total_length(self) -> int:
        """Total customers waiting (VIP + normal)."""
        return len(self.queue) + len(self.priority_heap)

    def estimated_wait_time(self) -> float:
        """
        Waiting Time Estimate (DSA formula):
          wait = queue_length × avg_service_time
        """
        return self.total_length() * self.avg_service_time

    def peek_next(self):
        """Who is at the front? (without removing)"""
        if self.priority_heap:
            return self.priority_heap[0][1]
        if self.queue:
            return self.queue[0]
        return None

    def all_customers(self):
        """Return a flat list of all waiting customers (for display)."""
        vip_list  = [c for (_, c) in sorted(self.priority_heap)]
        norm_list = list(self.queue)
        return vip_list + norm_list

    def __repr__(self):
        return (f"Counter-{self.counter_id} | "
                f"Queue={self.total_length()} | "
                f"Wait≈{self.estimated_wait_time():.1f}s")


# ─────────────────────────────────────────────
# QueueManager — manages ALL counters
# ─────────────────────────────────────────────
class QueueManager:
    """
    Top-level manager holding multiple ServiceCounters.
    Exposes simple API used by scheduler and UI.
    """

    def __init__(self, num_counters: int = 4, avg_service_time: int = 3):
        self.counters = [
            ServiceCounter(i + 1, avg_service_time)
            for i in range(num_counters)
        ]
        self.event_log = []          # list of strings describing what happened
        self.history   = []          # (tick, [queue_lengths]) for the graph

    # ── Logging helper ────────────────────────
    def log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > 100:          # cap log size
            self.event_log.pop(0)

    # ── Snapshot for graph ────────────────────
    def record_snapshot(self):
        lengths = [c.total_length() for c in self.counters]
        self.history.append(lengths)
        if len(self.history) > 50:             # keep last 50 ticks
            self.history.pop(0)

    # ── Simple getters ────────────────────────
    def get_counter(self, counter_id: int) -> ServiceCounter:
        return self.counters[counter_id - 1]

    def total_customers(self) -> int:
        return sum(c.total_length() for c in self.counters)

    def all_queue_lengths(self) -> list[int]:
        return [c.total_length() for c in self.counters]
