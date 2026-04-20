"""
main.py
-------
Entry point for the AI-Inspired Dynamic Crowd & Service Flow Optimizer.

Run the app with:
    streamlit run main.py

Or run a headless simulation (no UI) with:
    python main.py --simulate
"""

import sys
import os

# ── Make sure imports work regardless of working directory ──
sys.path.insert(0, os.path.dirname(__file__))


def run_headless_demo():
    """
    CLI demo — shows the DSA logic working without Streamlit.
    Useful for understanding what's happening under the hood.
    """
    from queue_manager import QueueManager, Customer
    from scheduler import Scheduler
    from simulation import (
        simulate_single_arrival,
        simulate_arrival_burst,
        seed_initial_customers,
    )

    print("=" * 60)
    print("  AI Queue Optimizer — Headless Demo")
    print("=" * 60)

    # 1. Setup
    manager   = QueueManager(num_counters=4, avg_service_time=3)
    scheduler = Scheduler(manager)

    # 2. Seed initial customers
    print("\n[1] Seeding 8 initial customers...")
    seed_initial_customers(scheduler, num_customers=8, vip_probability=0.3)

    for c in manager.counters:
        print(f"   {c}")

    # 3. Serve some customers
    print("\n[2] Serving 3 customers...")
    for _ in range(3):
        cust, cid = scheduler.serve_next_global()
        if cust:
            print(f"   Served: {cust} from Counter-{cid}")

    # 4. Trigger rebalance
    print("\n[3] Rebalancing queues...")
    scheduler.rebalance()

    # 5. Add a burst
    print("\n[4] Burst arrival of 5 customers...")
    simulate_arrival_burst(scheduler, count=5, vip_probability=0.2)

    # 6. Emergency injection
    print("\n[5] Emergency injection...")
    cid = scheduler.inject_emergency("Dr. Sharma")
    print(f"   Emergency patient injected into Counter-{cid}")

    # 7. Status
    print("\n[6] Final status:")
    status = scheduler.status_summary()
    print(f"   Mode      : {status['mode']}")
    print(f"   Peak      : {status['is_peak']}")
    print(f"   Total in Q: {status['total_customers']}")
    print(f"   Lengths   : {status['queue_lengths']}")
    print(f"   Wait times: {[f'{w:.1f}s' for w in status['wait_times']]}")

    # 8. Print log
    print("\n[7] Event log (last 10):")
    for entry in manager.event_log[-10:]:
        print(f"   {entry}")

    print("\n" + "=" * 60)
    print("  Demo complete. Run `streamlit run main.py` for the UI.")
    print("=" * 60)


def run_streamlit():
    """Launch the Streamlit UI."""
    import subprocess
    subprocess.run(
        ["streamlit", "run", os.path.join(os.path.dirname(__file__), "ui.py")],
        check=True
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        run_headless_demo()
    else:
        # Default: print instructions (streamlit run handles the UI directly)
        print("\n🚀  AI Queue Optimizer")
        print("─" * 40)
        print("To launch the dashboard:")
        print("    streamlit run ui.py")
        print("\nTo run the headless CLI demo:")
        print("    python main.py --simulate")
        print("─" * 40)
        run_headless_demo()
