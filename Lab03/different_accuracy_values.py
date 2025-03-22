import subprocess
import re
import matplotlib.pyplot as plt
import math

# Function to run the executable and provide input
def run_executable(executable, accuracy, radius, input2):
    process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_data = f"{accuracy}\n{radius}\n{input2}\n"
    stdout, stderr = process.communicate(input=input_data)
    return stdout

# Function to parse the relevant lines from stdout
def parse_output(stdout):
    # Regex patterns to extract success status and accuracy
    success_pattern = r"The program is correct"
    accuracy_pattern = r"The accuracy based on the worse performing difference is: ([\d.]+)"
    
    success = bool(re.search(success_pattern, stdout))
    accuracy_match = re.search(accuracy_pattern, stdout)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    return success, accuracy

# Function to convert an accuracy value to its decimal precision
def accuracy_to_decimals(accuracy):
    if accuracy >= 1:
        return 0  # No decimals for accuracy >= 1
    return -int(math.log10(accuracy))  # Logarithmic base-10 for precision

# Main function
def main():
    executable = "modification_3/Convolution2D"  # Path to your executable
    input2 = 1024  # Fixed third input parameter
    # radius_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Radius values
    radius_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Radius values

    final_accuracies = []  # To store the accuracy for each radius

    for radius in radius_values:
        print(f"Processing radius: {radius}")
        accuracy = 0.00000000000000000001 # Initial accuracy
        success = False

        # Double accuracy until the program is successful
        while not success:
            stdout = run_executable(executable, accuracy, radius, input2)
            success, reported_accuracy = parse_output(stdout)

            if not success:
                accuracy *= 2  # Double the accuracy
                print(f"Accuracy increased to {accuracy} for radius {radius}")

        final_accuracies.append(accuracy)
        print(f"Final Accuracy for radius {radius}: {accuracy}")
        print("-" * 50)

    # Convert accuracies to decimal precision
    decimals = [accuracy_to_decimals(acc) for acc in final_accuracies]

    # Create an index for equal spacing
    indices = list(range(len(radius_values)))  # Indices for equal spacing

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(indices, decimals, marker='o', color="orange", label="Accuracy (Decimals)")
    
    # Annotate the points with the corresponding decimal values
    for idx, decimal in zip(indices, decimals):
        plt.annotate(f"{decimal}", (idx, decimal), textcoords="offset points", xytext=(0, 10), ha='center', color="orange")

    # Customize x-axis with labels corresponding to radius values
    plt.xticks(indices, radius_values)  # Set the ticks to indices and labels to radius values
    plt.xlabel("Radius Values")
    plt.ylabel("Accuracy (Decimals)")
    plt.title("Final Accuracy (Decimal Precision) for Each Radius Value")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()