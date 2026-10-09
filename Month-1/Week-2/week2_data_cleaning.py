"""Skybrisk Month 1 - Week 2: Python data structures, functions, and data cleaning."""

from statistics import mean

def clean_temperatures(values):
    """Remove None and blank entries, then convert valid values to floats."""
    cleaned = []
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            print(f"Skipping invalid temperature: {value!r}")
    return cleaned

def summarize(values):
    cleaned = clean_temperatures(values)
    if not cleaned:
        return {"count": 0, "average": None, "minimum": None, "maximum": None}
    return {
        "count": len(cleaned),
        "average": mean(cleaned),
        "minimum": min(cleaned),
        "maximum": max(cleaned),
    }

if __name__ == "__main__":
    raw_temperatures = [29.5, 30, None, 28.5, "", 31, "invalid", 29]
    print("Raw temperatures:", raw_temperatures)
    print("Cleaned temperatures:", clean_temperatures(raw_temperatures))
    print("Summary:", summarize(raw_temperatures))
    print("List comprehension:", [x * 2 for x in [1, 2, 3]])
    print("Dictionary:", {"city": "Hyderabad", "temperature": 30})
