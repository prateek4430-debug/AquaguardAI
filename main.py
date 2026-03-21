import pandas as pd
import numpy as np

SEED = 7
TOTAL_SAMPLES = 500
np.random.seed(SEED)

WHO_PH_LOW = 6.5
WHO_PH_HIGH = 8.5
WHO_TURBIDITY = 5.0
WHO_TDS = 600
WHO_NITRATE = 45
WHO_CHLORINE_LOW = 0.2
WHO_CHLORINE_HIGH = 1.2
WHO_HARDNESS = 300

def make_water_data(size):
    ph_vals      = np.random.normal(7.0, 0.8, size).clip(4.5, 9.5)
    turb_vals    = np.random.exponential(3, size).clip(0.1, 20)
    tds_vals     = np.random.normal(350, 120, size).clip(50, 900)
    hard_vals    = np.random.normal(180, 60, size).clip(50, 400)
    chlor_vals   = np.random.uniform(0.1, 1.5, size)
    nitrate_vals = np.random.exponential(15, size).clip(1, 80)
    temp_vals    = np.random.normal(25, 5, size).clip(10, 40)
    bact_vals    = np.random.choice([0, 1], size, p=[0.75, 0.25])

    labels = []
