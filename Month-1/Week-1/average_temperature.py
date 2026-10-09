
# Skybrisk Internship - Month 1, Week 1
# Average Temperature Calculator

# Store daily temperature readings in a list
temperatures = [29.5, 30.0, 28.5, 31.0, 29.0]

# Check whether temperature readings are available
if len(temperatures) > 0:

    # Calculate the total of all readings
    total = sum(temperatures)

    # Calculate the average temperature
    average = total / len(temperatures)

    # Display the results
    print("Daily temperatures:", temperatures)
    print("Number of readings:", len(temperatures))
    print(f"Total temperature: {total:.2f}°C")
    print(f"Average temperature: {average:.2f}°C")

else:
    print("No temperature readings are available.")