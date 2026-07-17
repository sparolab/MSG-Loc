import numpy as np

def print_error_stats(error_array, TOTAL_FRAMES=0):
    num_success = len(error_array)
    if TOTAL_FRAMES and TOTAL_FRAMES > 0:
        if TOTAL_FRAMES < num_success:
            print(f"[Warning] Specified TOTAL_FRAMES({TOTAL_FRAMES}) is less than "
                  f"the number of successful frames({num_success}). Using "
                  "max(TOTAL_FRAMES, successful frames) as the denominator.")
        denom = max(TOTAL_FRAMES, num_success)  # Denominator for ratio calculations
        failed = max(TOTAL_FRAMES - num_success, 0)
    else:
        denom = num_success
        failed = 0

    total_count = num_success + failed

    print("="*40)
    print(f"TOTAL_FRAMES = {TOTAL_FRAMES}")
    print("-"*40)
    print("\nArray Shape:", error_array.shape)
    print("Array Data Type:", error_array.dtype)

    print("\nArray Statistics:")
    print("Mean:", np.mean(error_array))
    print("Std Dev:", np.std(error_array))
    print("Max:", np.max(error_array))
    print("Min:", np.min(error_array))

    thresholds = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    print("\nError Distribution by Thresholds:")
    for threshold in thresholds:
        count_in_range = np.sum(error_array <= threshold)
        ratio_in_range = count_in_range / total_count
        print(f"@{threshold}: Count = {count_in_range}, Ratio = {ratio_in_range:.4f}")

    percentiles = [10, 20, 50]
    print("\nBottom Percentile Values (Lower is Better):")
    for percentile in percentiles:
        value = np.percentile(error_array, percentile)
        count_below = np.sum(error_array <= value)
        ratio_below = count_below / total_count
        print(f"Bottom {percentile}%: Value <= {value:.4f}, Count = {count_below}, Ratio = {ratio_below:.4f}")

    print(f"\nTotal Number of Errors: {total_count}")
    print("="*40 + "\n")


# ---- Usage example ----

# Load the error_array file
error_array = np.load('/root/workspace/results/error_array.npy')

# Set the desired TOTAL_FRAMES value (e.g., 0 and 3007, 2964, 4067, 1282, 1309, 1596, or 1640)
# dji: 2001 / ground: 1600 / vr: 2083
for TOTAL_FRAMES in [0, 2964]:
    print_error_stats(error_array, TOTAL_FRAMES=TOTAL_FRAMES)
