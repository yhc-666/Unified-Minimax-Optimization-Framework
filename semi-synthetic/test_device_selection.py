#!/usr/bin/env python
"""
Test script to verify device selection functionality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evaluation import validate_device

def test_device_validation():
    """Test the validate_device function."""

    print("Testing device validation...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Test 1: Auto-detect (None)
    print("Test 1: Auto-detect (device=None)")
    device = validate_device(None)
    print(f"  Result: {device}")
    print()

    # Test 2: CPU
    print("Test 2: Explicit CPU (device='cpu')")
    device = validate_device('cpu')
    print(f"  Result: {device}")
    print()

    # Test 3: CUDA (if available)
    if torch.cuda.is_available():
        print("Test 3: Default CUDA (device='cuda')")
        device = validate_device('cuda')
        print(f"  Result: {device}")
        print()

        print("Test 4: Specific GPU (device='cuda:0')")
        device = validate_device('cuda:0')
        print(f"  Result: {device}")
        print()

        if torch.cuda.device_count() > 1:
            print("Test 5: Second GPU (device='cuda:1')")
            device = validate_device('cuda:1')
            print(f"  Result: {device}")
            print()

        # Test 6: Invalid GPU index
        print(f"Test 6: Invalid GPU index (device='cuda:{torch.cuda.device_count()}')")
        try:
            device = validate_device(f'cuda:{torch.cuda.device_count()}')
            print(f"  ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"  Correctly raised ValueError: {e}")
        print()
    else:
        # Test 7: CUDA when not available
        print("Test 7: CUDA when not available (device='cuda')")
        try:
            device = validate_device('cuda')
            print(f"  ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"  Correctly raised ValueError: {e}")
        print()

    print("All tests completed!")

if __name__ == "__main__":
    test_device_validation()
