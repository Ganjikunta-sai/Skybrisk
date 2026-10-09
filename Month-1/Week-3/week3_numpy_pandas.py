"""Skybrisk Month 1 - Week 3: NumPy arrays and Pandas DataFrames."""

import numpy as np
import pandas as pd

def main():
    temperatures = np.array([29.5, 30.0, 28.5, 31.0, 29.0])
    print("NumPy array:", temperatures)
    print("Broadcasting: add 1 degree to every value:", temperatures + 1)
    print("Average:", temperatures.mean())
    print("Minimum:", temperatures.min())
    print("Maximum:", temperatures.max())

    df = pd.DataFrame({
        "City": ["Hyderabad", "Chennai", "Hyderabad", "Chennai", "Bengaluru"],
        "Day": ["Mon", "Mon", "Tue", "Tue", "Mon"],
        "Temperature": [29.5, 32.0, 30.0, 33.0, 26.5],
    })
    print("\nFirst rows:\n", df.head())
    print("\nHyderabad records:\n", df[df["City"] == "Hyderabad"])
    print("\nAverage temperature by city:\n", df.groupby("City")["Temperature"].mean())

if __name__ == "__main__":
    main()
