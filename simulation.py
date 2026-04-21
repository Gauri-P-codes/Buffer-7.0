"""
simulation.py
-------------
Event-driven simulation engine.
"""
import random
import time
from queue_manager import QueueManager, Customer
from scheduler import Scheduler

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Meera", "Arjun", "Sneha",
    "Vikram", "Ananya", "Karan", "Diya", "Ravi", "Pooja",
    "Aditya", "Nisha", "Kabir", "Simran", "Raj", "Isha",
    "Dev", "Tanya", "Sam", "Rita", "Leo", "Maya", "Zara",
]

def random_customer(vip_probability: float = 0.2) -> Customer:
    name         = random.choice(FIRST_NAMES)
    is_vip       = random.random() < vip_probability
    service_time = random.randint(2, 8)
    return Customer(name, is_vip=is_vip, service_time=service_time)

def simulate_arrival_burst(
    scheduler: Scheduler,
    count: int = 5,
    vip_probability: float = 0.2
) -> list[Customer]:
    added = []
    for i in range(count):
        customer = random_customer(vip_probability)
        customer.arrival_time = time.time() + i * 0.001  # distinct FCFS timestamps
        scheduler.assign_customer(customer)
        added.append(customer)
    return added

def simulate_single_arrival(
    scheduler: Scheduler,
    is_vip: bool = False,
    name: str | None = None
) -> Customer:
    cust_name    = name or random.choice(FIRST_NAMES)
    service_time = random.randint(2, 8)
    customer     = Customer(cust_name, is_vip=is_vip, service_time=service_time)
    scheduler.assign_customer(customer)
    return customer

def run_auto_simulation_step(
    scheduler: Scheduler,
    arrival_prob: float = 0.6,
    serve_prob: float   = 0.5,
    vip_prob: float     = 0.15
):
    events = []
    if random.random() < arrival_prob:
        customer = random_customer(vip_probability=vip_prob)
        scheduler.assign_customer(customer)
        events.append(f"Arrived: {customer}")
    if random.random() < serve_prob:
        customer, counter_id = scheduler.serve_next_global()
        if customer:
            events.append(f"Served: {customer} from Counter-{counter_id}")
    scheduler.manager.record_snapshot()
    return events

def seed_initial_customers(
    scheduler: Scheduler,
    num_customers: int = 8,
    vip_probability: float = 0.25
):
    for i in range(num_customers):
        customer = random_customer(vip_probability)
        customer.arrival_time = time.time() + i * 0.001  # distinct FCFS timestamps
        scheduler.assign_customer(customer)
    scheduler.manager.log(
        f"🌱 System seeded with {num_customers} initial customers."
    )
    scheduler.manager.record_snapshot()
