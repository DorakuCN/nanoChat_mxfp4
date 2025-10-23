#!/usr/bin/env python3
"""
Test script to verify wandb integration
"""
import os
import sys
import time
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_config import setup_wandb, log_metrics, finish_wandb

def test_wandb_integration():
    """Test wandb integration with a simple run"""
    print("Testing wandb integration...")
    
    # Test configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model_type": "test_model",
        "test_run": True
    }
    
    # Setup wandb
    run_name = f"test_run_{int(time.time())}"
    wandb_run = setup_wandb(run_name=run_name, config=config, tags=["test", "integration"])
    
    if hasattr(wandb_run, 'log'):
        print("✓ Wandb integration successful!")
        
        # Log some test metrics
        for step in range(10):
            metrics = {
                "loss": random.uniform(0.1, 1.0),
                "accuracy": random.uniform(0.8, 0.95),
                "learning_rate": config["learning_rate"] * (0.9 ** step)
            }
            log_metrics(metrics, step=step)
            print(f"Logged metrics for step {step}: {metrics}")
            time.sleep(0.1)
        
        # Finish the run
        finish_wandb()
        print("✓ Test run completed successfully!")
        
    else:
        print("⚠ Wandb is disabled (using DummyWandb)")
        print("To enable wandb, make sure:")
        print("1. WANDB_MODE is not set to 'disabled'")
        print("2. WANDB_RUN is not set to 'dummy'")
        print("3. wandb is properly logged in")

if __name__ == "__main__":
    test_wandb_integration()
