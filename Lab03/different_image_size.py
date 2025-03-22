import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to run the executable and provide input
def run_executable(executable, radius, image_size):
    process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_data = f"0.0001\n{radius}\n{image_size}\n"  # Fixed accuracy of 0.001
    stdout, stderr = process.communicate(input=input_data)
    return stdout

# Function to parse the relevant lines from stdout
def parse_output(stdout):
    # Regex patterns to extract CPU and GPU times
    cpu_pattern = r"CPU Execution time: ([\d.]+) seconds"
    gpu_pattern = r"GPU Execution time: ([\d.]+) seconds"

    cpu_match = re.search(cpu_pattern, stdout)
    gpu_match = re.search(gpu_pattern, stdout)

    cpu_time = float(cpu_match.group(1)) if cpu_match else None
    gpu_time = float(gpu_match.group(1)) if gpu_match else None

    return cpu_time, gpu_time

# Function to calculate the average while removing outliers
def calculate_average(values):
    if len(values) < 3:
        return np.mean(values)  # Not enough data to remove outliers
    sorted_values = sorted(values)
    trimmed_values = sorted_values[1:-1]  # Remove min and max
    return np.mean(trimmed_values)

# Main function
def main():
    executable = "modification_3/Convolution2D"  # Path to your executable
    radius = 16  # Fixed radius value
    image_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # Different image sizes to test

    avg_cpu_times = []
    avg_gpu_times = []

    for image_size in image_sizes:
        print(f"Processing image size: {image_size}")
        cpu_times = []
        gpu_times = []

        # Run 12 iterations for each image size
        for _ in range(12):
            stdout = run_executable(executable, radius, image_size)
            cpu_time, gpu_time = parse_output(stdout)

            if cpu_time is not None:
                cpu_times.append(cpu_time)
            if gpu_time is not None:
                gpu_times.append(gpu_time)

        # Calculate averages after removing outliers
        avg_cpu_time = calculate_average(cpu_times) if cpu_times else None
        avg_gpu_time = calculate_average(gpu_times) if gpu_times else None

        avg_cpu_times.append(avg_cpu_time)
        avg_gpu_times.append(avg_gpu_time)

        print(f"Image size {image_size}: Average CPU Time = {avg_cpu_time:.4f} s, Average GPU Time = {avg_gpu_time:.4f} s")
        print("-" * 50)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(image_sizes, avg_cpu_times, marker='o', color="blue", label="CPU Time")
    plt.plot(image_sizes, avg_gpu_times, marker='s', color="red", label="GPU Time")

    # Annotate the points with runtime values
    for x, cpu, gpu in zip(image_sizes, avg_cpu_times, avg_gpu_times):
        plt.annotate(f"{cpu:.2f}s", (x, cpu), textcoords="offset points", xytext=(0, 10), ha='center', color="blue")
        plt.annotate(f"{gpu:.2f}s", (x, gpu), textcoords="offset points", xytext=(0, -15), ha='center', color="red")

    plt.xlabel("Image Size")
    plt.ylabel("Runtime (seconds)")
    plt.title("CPU and GPU Runtimes for Different Image Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
