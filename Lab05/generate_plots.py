import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparisons_with_secondary_axis(excel_file, output_dir):
    # Load the Excel file
    data = pd.read_excel(excel_file)

    # Fill NaN values in the "Path" column with the previous value
    data["Path"].fillna(method="ffill", inplace=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique paths and bodies
    paths = data["Path"].unique()
    bodies = data["Bodies"].unique()

    # Iterate through paths (current and previous)
    for i in range(1, len(paths)):  # Start from index 1 to compare with the previous path
        current_path = paths[i]
        previous_path = paths[i - 1]

        # Filter data for the current and previous paths
        current_data = data[data["Path"] == current_path]
        previous_data = data[data["Path"] == previous_path]

        # Prepare data for plotting
        avg_time_previous = []
        avg_time_current = []
        std_dev_previous = []
        std_dev_current = []
        body_labels = []

        for body in bodies:
            # Filter data for the specific body
            current_body_data = current_data[current_data["Bodies"] == body]
            previous_body_data = previous_data[previous_data["Bodies"] == body]

            # Check if both current and previous data exist for the body
            if current_body_data.empty or previous_body_data.empty:
                continue

            # Extract Average Time and Std Dev Time
            avg_time_previous.append(previous_body_data["Average Time (s)"].values[0])
            avg_time_current.append(current_body_data["Average Time (s)"].values[0])
            std_dev_previous.append(previous_body_data["Std Dev Time (s)"].values[0])
            std_dev_current.append(current_body_data["Std Dev Time (s)"].values[0])
            body_labels.append(body)

        # Plot combined comparison for all bodies
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)  # Higher resolution

        bar_width = 0.25  # Thinner bars
        x = range(len(body_labels))  # Position of groups on the x-axis

        # Plot bars on the primary axis
        bars_previous = ax1.bar([pos - bar_width / 2 for pos in x], avg_time_previous, width=bar_width, 
                                label=f"{previous_path} - Avg Time", color="lightyellow", edgecolor="black")
        bars_current = ax1.bar([pos + bar_width / 2 for pos in x], avg_time_current, width=bar_width, 
                                label=f"{current_path} - Avg Time", color="lightgreen", edgecolor="black")

        # Add values on top of bars in scientific notation
        for bar, value in zip(bars_previous, avg_time_previous):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2e}", 
                     ha='center', va='bottom', fontsize=9, color="black")
        for bar, value in zip(bars_current, avg_time_current):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2e}", 
                     ha='center', va='bottom', fontsize=9, color="black")

        # Plot standard deviation as lines on the secondary axis
        ax2 = ax1.twinx()  # Create a secondary y-axis
        ax2.plot(x, std_dev_previous, label=f"{previous_path} - Std Dev", color="orange", marker="o", linestyle="-")
        ax2.plot(x, std_dev_current, label=f"{current_path} - Std Dev", color="purple", marker="o", linestyle="-")

        # Customize axes
        ax1.set_xlabel("Bodies", fontsize=12)
        ax1.set_ylabel("Average Time (s)", fontsize=12)
        ax2.set_ylabel("Standard Deviation (s)", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(body_labels, fontsize=10)

        # Add legends
        bar_legend = ax1.legend(loc="upper left", fontsize=10)
        std_dev_legend = ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=10)

        # Add title
        plt.title(f"Comparison Between {previous_path} and {current_path}", fontsize=14)

        # Save plot
        plot_filename = os.path.join(output_dir, f"Comparison_{previous_path}_vs_{current_path}.png")
        plt.savefig(plot_filename, bbox_inches="tight")
        print(f"Plot saved: {plot_filename}")
        plt.close()

# Example usage
excel_file = "Runtime_results_v2.xlsx"
output_dir = "output_plots"
plot_comparisons_with_secondary_axis(excel_file, output_dir)
