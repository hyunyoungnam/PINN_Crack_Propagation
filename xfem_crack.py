#!/usr/bin/env python3
"""
Backward-compatible entry point for X-PINN fracture mechanics.
The full implementation is in the xpinn_fracture package.
Run: python xfem_crack.py
  or: python run_experiments.py  (full experiment with validation)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from run_experiments import main

if __name__ == "__main__":
    main()
