import numpy as np
import time
import os

from heal_swin.utils.get_paths import get_syn_datasets_path


use_masking = False
data_transf = "inv"


def mask_inverse(mask):
    return 1 / mask


def id(mask):
    return mask


data_transfs = {
    "log": np.log,
    "inv": mask_inverse,
    "None": id,
}

mask_dir = os.path.join(get_syn_datasets_path(), "depth_maps", "raw_data")

depth_masks = os.listdir(mask_dir)

full_list = []

prev_shape = None
prev_mask = None

masks = {}

total_masks = len(depth_masks)

all_data = np.empty((0,))

batch_size = 50
batch_nr = -1
curr_ind = -1
completed = False
batches = {}

total_start_time = time.time()
num_infs = 0
total_pixels = 0
while not completed:
    batch_nr += 1
    start = curr_ind + 1
    end = start + batch_size
    batches[batch_nr] = np.empty((0,))
    batch_start_time = time.time()
    for ind in range(start, min(end, total_masks)):
        start_time = time.time()
        mask = np.load(os.path.join(mask_dir, depth_masks[ind])).flatten()

        idcs = mask != 1000  # Get non inf pixels
        num_infs += np.count_nonzero(~idcs)  # Negate and count for total infs
        total_pixels += mask.size

        if use_masking:
            mask = mask[idcs]
        mask = data_transfs[data_transf](mask)

        batches[batch_nr] = np.concatenate((batches[batch_nr], mask))
        end_time = time.time()
        delta = end_time - start_time
        print(
            f"batch {batch_nr}: mask {ind+1} of {total_masks} loaded in {delta:.4f}s...", end="\r"
        )
        if ind + 1 == total_masks:
            completed = True
            break
    batch_end_time = time.time()
    batch_total_time = batch_end_time - batch_start_time

    print(
        f"batch {batch_nr}: mask {ind+1} of {total_masks} loaded in {delta:.4f}s... total:",
        f"{batch_total_time:.4f}s",
    )
    curr_ind = ind

end_load_time = time.time()
delta_load_time = end_load_time - total_start_time
print(f"loading time of {ind+1} masks was {delta_load_time:.4f}s")

print("concatenating batches... ")
all_data = np.empty((0,))
for key in batches.keys():
    print("concatenating batch nr ", key, end="\r")
    all_data = np.concatenate((all_data, batches[key]))

print("concatenating batch nr ", key, end=" ")
print("done")

end_concat_time = time.time()
delta_concat_time = end_concat_time - end_load_time

print(f"final concat of {batch_nr+1} batches a {batch_size} masks took {delta_concat_time:.4f}s")
total_end_time = time.time()
total_delta = total_end_time - total_start_time

print(f"total time of {ind+1} masks took {total_delta:.4f}s")

max = np.amax(all_data)
min = np.amin(all_data)
mean = np.mean(all_data)
std = np.std(all_data)


print(
    "max of all non background pixels (only relevant when masking is not used): ",
    np.amax(all_data[all_data != 1000]),
)
print("max : ", max)
print("min: ", min)
print("mean: ", mean)
print("std: ", std)
print("")
print("total pixels: ", total_pixels)
print("total background: ", num_infs)

hist_details = np.histogram(all_data, bins=1000)
np.savez(
    os.path.join(
        get_syn_datasets_path(),
        f"depth_histogram_masked_{use_masking}_data_transformation_{data_transf}" + ".npz",
    ),
    bin_counts=hist_details[0],
    bin_edges=hist_details[1],
)

with open(f"depth_stats_masked_{use_masking}_data_transformation_{data_transf}.txt", "w") as file1:
    # Writing data to a file
    file1.write("data_transf: " + data_transf)
    file1.write("use_maskgin: " + str(use_masking))
    file1.write(
        "max of all non background pixels (only relevant when masking is not used): "
        + str(np.amax(all_data[all_data != 1000]))
    )
    file1.write("max : " + str(max))
    file1.write("min: " + str(min))
    file1.write("mean: " + str(mean))
    file1.write("std: " + str(std))
    file1.write("")
    file1.write("total pixels: " + str(total_pixels))
    file1.write("total background: " + str(num_infs))
