import sys

def parse_line(line):
    """Parses a line and extracts the body ID and values."""
    parts = line.strip().split(", ")
    body_id = int(parts[0].split()[1][:-1])  # Extract the ID
    values = {}
    for part in parts[1:]:
        key, value = part.split("=")
        values[key] = float(value)
    return body_id, values

def compare_files(file1, file2, threshold=0.01):
    """
    Compares two files line by line.
    Reports differences exceeding the threshold for any value.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            print("Files have different numbers of lines.")
            return

        differences_found = False
        for line1, line2 in zip(lines1, lines2):
            body_id1, values1 = parse_line(line1)
            body_id2, values2 = parse_line(line2)

            if body_id1 != body_id2:
                print(f"Body ID mismatch: {body_id1} != {body_id2}")
                differences_found = True
                continue

            for key in values1.keys():
                diff = abs(values1[key] - values2[key])
                if diff > threshold:
                    print(f"Body {body_id1}, {key} differs: {values1[key]} vs {values2[key]} (diff={diff:.3f})")
                    differences_found = True

        if not differences_found:
            print("No differences found beyond the threshold.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    threshold = 0.005  # Threshold for differences beyond the 2nd decimal place

    compare_files(file1, file2, threshold)
