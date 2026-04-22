
import heapq
from queue_manager import QueueManager, Customer
from scheduler import Scheduler

def test_sjf_fix():
    manager = QueueManager(num_counters=2)
    scheduler = Scheduler(manager)
    
    # Same service time for customers in different counters
    cust1 = Customer("Alice", service_time=5)
    cust2 = Customer("Bob", service_time=5)
    
    manager.get_counter(1).add_customer(cust1)
    manager.get_counter(2).add_customer(cust2)
    
    scheduler.mode = "SJF"
    
    try:
        # This will call _serve_sjf which uses heapq with candidates
        served, cid = scheduler.serve_next_global()
        print(f"Success: Served {served.name} from Counter-{cid}")
        
        served, cid = scheduler.serve_next_global()
        print(f"Success: Served {served.name} from Counter-{cid}")
        
    except TypeError as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_sjf_fix()
