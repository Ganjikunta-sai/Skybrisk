"""Skybrisk Month 1 - Week 4: Visualize data with Matplotlib and Seaborn."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.DataFrame({
        "City": ["Hyderabad", "Chennai", "Bengaluru", "Hyderabad", "Chennai", "Bengaluru"],
        "Day": ["Mon", "Mon", "Mon", "Tue", "Tue", "Tue"],
        "Temperature": [29.5, 32.0, 26.5, 30.0, 33.0, 27.0],
        "Humidity": [55, 68, 60, 52, 70, 58],
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="City", y="Temperature", errorbar=None)
    plt.title("Temperature by City")
    plt.tight_layout()
    plt.savefig("average_temperature_by_city.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Temperature", y="Humidity", hue="City", s=100)
    plt.title("Temperature vs Humidity")
    plt.tight_layout()
    plt.savefig("temperature_vs_humidity.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["Temperature", "Humidity"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Temperature and Humidity Correlation")
    plt.tight_layout()
    plt.savefig("temperature_humidity_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
