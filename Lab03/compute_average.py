import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np  # For calculating mean and removing outliers

# Function to run the executable and provide input
def run_executable(executable, input1, input2):
    process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_data = f"{input1}\n{input2}\n"
    stdout, stderr = process.communicate(input=input_data)
    return stdout

# Function to parse the relevant lines from stdout
def parse_output(stdout):
    # Regex patterns to extract CPU time, GPU time, and accuracy
    cpu_pattern = r"CPU Execution time: ([\d.]+) seconds"
    gpu_pattern = r"GPU Execution time: ([\d.]+) seconds"
    accuracy_pattern = r"The accuracy based on the worse performing difference is: ([\d.]+)"

    cpu_match = re.search(cpu_pattern, stdout)
    gpu_match = re.search(gpu_pattern, stdout)
    accuracy_match = re.search(accuracy_pattern, stdout)

    cpu_time = float(cpu_match.group(1)) if cpu_match else None
    gpu_time = float(gpu_match.group(1)) if gpu_match else None
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    return cpu_time, gpu_time, accuracy

# Function to calculate the average while removing outliers
def calculate_average(values):
    if len(values) < 3:  # Not enough data to remove outliers
        return np.mean(values)
    sorted_values = sorted(values)
    trimmed_values = sorted_values[1:-1]  # Remove min and max
    return np.mean(trimmed_values)

# Main function
def main():
    executable = "modification_1/Convolution2D"  # Path to your executable
    input2 = 32   # Fixed first input parameter
    param_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Different values for the second parameter
    specific_values = [32]  # The second input values you want to annotate

    avg_accuracies = []
    avg_cpu_times = []
    avg_gpu_times = []

    for input1 in param_values:
        print(f"Running for input1={input1}")
        accuracies = []
        cpu_times = []
        gpu_times = []

        # Run 12 iterations for each parameter
        for _ in range(12):
            stdout = run_executable(executable, input1, input2)
            cpu_time, gpu_time, accuracy = parse_output(stdout)

            if accuracy is not None:
                accuracies.append(accuracy)
            if cpu_time is not None:
                cpu_times.append(cpu_time)
            if gpu_time is not None:
                gpu_times.append(gpu_time)

        # Calculate averages after removing outliers
        avg_accuracy = calculate_average(accuracies) if accuracies else None
        avg_cpu_time = calculate_average(cpu_times) if cpu_times else None
        avg_gpu_time = calculate_average(gpu_times) if gpu_times else None

        avg_accuracies.append(avg_accuracy)
        avg_cpu_times.append(avg_cpu_time)
        avg_gpu_times.append(avg_gpu_time)

        print(f"Average Accuracy for input2={input2}: {avg_accuracy}")
        print(f"Average CPU Time for input2={input2}: {avg_cpu_time}")
        print(f"Average GPU Time for input2={input2}: {avg_gpu_time}")
        print("-" * 50)

    # Convert CPU and GPU times to milliseconds (multiply by 1000)
    avg_cpu_times_ms = [time * 1000 for time in avg_cpu_times]
    avg_gpu_times_ms = [time * 1000 for time in avg_gpu_times]

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy Plot
    ax1.set_xlabel("Radius Values")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(param_values, avg_accuracies, marker='o', color="tab:blue", label="Accuracy")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True)

    # Annotate specific values only if accuracy is not None
    for value in specific_values:
        if value in param_values:
            idx = param_values.index(value)
            if avg_accuracies[idx] is not None:
                ax1.annotate(f'{avg_accuracies[idx]:.4f}', 
                             (param_values[idx], avg_accuracies[idx]), 
                             textcoords="offset points", 
                             xytext=(0, 10), ha='center', color="tab:blue")

    # Runtime Plot
    ax2 = ax1.twinx()
    ax2.set_ylabel("Runtime (ms)", color="tab:red")  # Update label to ms
    ax2.plot(param_values, avg_cpu_times_ms, marker='s', linestyle='--', color="tab:red", label="CPU Time")
    ax2.plot(param_values, avg_gpu_times_ms, marker='^', linestyle='--', color="tab:green", label="GPU Time")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Add legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Make sure specific values show up on x-axis
    plt.xticks(param_values)  # Explicitly set x-ticks to the parameter values
    plt.title("Accuracy and Runtime Comparison for 32 image size and different radius values")
    plt.show()

if __name__ == "__main__":
    main()
