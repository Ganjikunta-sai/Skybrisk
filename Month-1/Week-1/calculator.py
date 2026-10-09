
# Skybrisk Internship - Month 1, Week 1
# Basic Calculator

# Take two numbers from the user
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

# Ask the user to select an operation
operation = input("Choose an operation (+, -, *, /): ").strip()

# Perform the selected calculation
if operation == "+":
    result = num1 + num2
    print("Result:", result)

elif operation == "-":
    result = num1 - num2
    print("Result:", result)

elif operation == "*":
    result = num1 * num2
    print("Result:", result)

elif operation == "/":
    if num2 == 0:
        print("Error: Cannot divide by zero.")
    else:
        result = num1 / num2
        print("Result:", result)

else:
    print("Invalid operation. Please choose +, -, *, or /.")