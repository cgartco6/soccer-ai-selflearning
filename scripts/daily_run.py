#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Pipeline

if __name__ == "__main__":
    pipe = Pipeline()
    pipe.daily_predict()
    print("Daily prediction completed.")
