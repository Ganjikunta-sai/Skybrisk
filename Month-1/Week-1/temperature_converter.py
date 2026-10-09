
# Skybrisk Internship - Month 1, Week 1
# Temperature Converter

print("Temperature Converter")
print("1. Celsius to Fahrenheit")
print("2. Fahrenheit to Celsius")

# Ask the user to choose a conversion
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    celsius = float(input("Enter temperature in Celsius: "))

    # Convert Celsius to Fahrenheit
    fahrenheit = (celsius * 9 / 5) + 32

    print(f"{celsius:.2f}°C = {fahrenheit:.2f}°F")

elif choice == "2":
    fahrenheit = float(input("Enter temperature in Fahrenheit: "))

    # Convert Fahrenheit to Celsius
    celsius = (fahrenheit - 32) * 5 / 9

    print(f"{fahrenheit:.2f}°F = {celsius:.2f}°C")

else:
    print("Invalid choice. Please enter 1 or 2.")