import matplotlib.pyplot as plt

# Data: Hardcoded values
image_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # Image sizes
cpu_times = [0.0004, 0.0016, 0.0073, 0.0353, 0.1418, 0.6272, 2.5911, 10.3249, 42.1773]  # CPU times in seconds
gpu_times = [0.0001, 0.0003, 0.0009, 0.0032, 0.0116, 0.0441, 0.1967, 0.6746, 2.4919]  # GPU times in seconds

# Plotting the data
def plot_data(image_sizes, cpu_times, gpu_times):
    plt.figure(figsize=(10, 6))

    # Generate evenly spaced indices for the image sizes
    indices = list(range(len(image_sizes)))

    # Plot CPU and GPU runtimes
    plt.plot(indices, cpu_times, marker='o', label="CPU Time", color="blue")
    plt.plot(indices, gpu_times, marker='s', label="GPU Time", color="red")

    # Annotating points with their values
    for i, cpu, gpu in zip(indices, cpu_times, gpu_times):
        plt.annotate(f"{cpu:.4f}s", (i, cpu), textcoords="offset points", xytext=(0, 10), ha='center', color="blue")
        plt.annotate(f"{gpu:.4f}s", (i, gpu), textcoords="offset points", xytext=(0, -15), ha='center', color="red")

    # Customize x-axis
    plt.xticks(indices, image_sizes)  # Set custom ticks and labels
    plt.xlabel("Image Size")
    plt.ylabel("Runtime (seconds)")
    plt.title("CPU and GPU Runtimes for Different Image Sizes")
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
def main():
    plot_data(image_sizes, cpu_times, gpu_times)

if __name__ == "__main__":
    main()
