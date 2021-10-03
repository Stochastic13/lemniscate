from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from colour import Color
import pickle
import re
import h5py


class JobPiece(object):  # redefining here (same code as batch_job)

    def __init__(self, prec, max_depth, bounding_box, cutoff, bigPictureID, max_piece):
        self.prec = prec
        self.max_iter = max_depth
        self.region = bounding_box
        self.cutoff = cutoff
        self.result = None
        self.bigPictureID = bigPictureID
        self.piece_size = max_piece

    def update_result(self, result):
        self.result = result.copy()

    def check_result(self):
        flat_array = self.result.flatten()
        min_val, max_val = np.min(flat_array), np.max(flat_array)
        assert min_val > 0, 'Zero in result. Array value not updated to iter value.'
        assert max_val <= self.max_iter, 'Iter value exceeds max_iter. Int overflow?'

    def blowup_stats(self, ax=None):
        flat_array = self.result.flatten()
        non_blowups = flat_array.tolist().count(self.max_iter)
        print('Percentage of blowups: ', (len(flat_array) - non_blowups) * 100 / len(flat_array))
        if ax is not None:
            ax.hist(flat_array, bins=100)


ani = None  # because the animation object must be assigned to a global variable for it to work


def analyze_cpu_log(path, anim=False, anim_frame_delay=1000, anim_per_frame=20):
    with open(path + '/cpu_log.log') as f:
        rd = f.read()
    rd = rd.split('\n\n')
    assert len(rd) == 2, 'More than one empty line in cpu_log'
    times_temps = rd[0]
    process_times = rd[1]
    times_temps = [x.split(':') for x in times_temps.split('\n')]
    relevant_cols = [i for i in range(len(times_temps[0])) if 'coreTime' in times_temps[0][i]]
    y_label = '$\\Delta$user CPU times'
    if len(relevant_cols) == 0:
        y_label = 'usage'
        relevant_cols = [i for i in range(len(times_temps[0])) if 'coreUsage' in times_temps[0][i]]
        times_only = np.array(times_temps[1:], dtype=np.float)
        times_only = times_only[:, np.array(relevant_cols, dtype=np.uint16)]
    else:
        # TODO: Use idle/system time too somewhere in the plot
        times_only = [[a.split(',')[0] if ',' in a else a for a in x] for x in times_temps[1:]]
        times_only = np.array(times_only, dtype=np.float)
        times_only = times_only[:, np.array(relevant_cols, dtype=np.uint16)]
        times_only = times_only - np.min(times_only, axis=0)
    assert len(relevant_cols) > 0, 'No CPU times/usage data found'
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel(y_label)
    ax.set_xlabel('update epochs')
    ax.set_title('CPU parameter ($\\Delta$ Time or Usage) over time (core-wise)')
    if not anim:
        # TODO: Plot the non-diffed version also somewhere
        main_line = []
        xvals = [x for x in range(times_only.shape[0] - 1)]
        for i in range(times_only.shape[1]):
            main_line.append(ax.plot(xvals, np.diff(times_only[:, i], axis=0), '^-', label=str(i)))
        ax.legend()
    else:
        main_line = []
        for i in range(times_only.shape[1]):
            line = ax.plot([], [], '^-', label=str(i))
            ax.set_ylim(0, np.max(np.diff(times_only, axis=0).flatten()) * 1.1)
            ax.set_xlim(0, anim_per_frame - 1)
            main_line.append(line)

        def ani_foo(frame, *args, **kwargs):
            ind = 0
            for line in main_line:
                line[0].set_data([i for i in range(frame.shape[0])], frame[:, ind].flatten())
                ind += 1
            return main_line

        global ani
        iterator = [np.diff(times_only[i:i + anim_per_frame + 1, :], axis=0) for i in
                    range(times_only.shape[1] - anim_per_frame)]
        ani = animation.FuncAnimation(fig, ani_foo, iterator, interval=anim_frame_delay, blit=False, repeat_delay=2000)
        plt.show()


def piece_together_large(path):
    indices_covered = set()

    def job_generator(path):
        files = [x for x in os.listdir(path) if x.isdigit()]
        for i in files:
            with open(path + '/' + i, 'rb') as f:
                rd = f.read()
            yield pickle.loads(rd)

    f = open(path + '/0', 'rb')
    temp = pickle.loads(f.read())
    f.close()
    ny, nx = temp.bigPictureID.split(':')[2:]
    piece_size = int(temp.piece_size)
    max_iter = int(temp.max_iter)
    upper_n = int(np.ceil(np.log2(max_iter)))
    upper_n = ((np.array([8, 16, 32, 64]) - upper_n) <= 0).tolist().index(False)
    upper_n = [8, 16, 32, 64][upper_n]
    sizes = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    del temp
    gen = job_generator(path)
    main_array = np.zeros((piece_size * int(ny), piece_size * int(nx)), dtype=sizes[upper_n])
    for piece in gen:
        piece.check_result()
        piece_id = [int(x) for x in piece.bigPictureID.split(':')[:2]]
        if piece.bigPictureID in indices_covered:
            assert False, 'Repeated bigPictureID'
        indices_covered.add(piece.bigPictureID)
        location = piece_id[0] * piece_size, piece_id[1] * piece_size
        main_array[location[0]:location[0] + piece_size, location[1]:location[1] + piece_size] = piece.result
        del piece
    return main_array


def piece_together_c_zoom(path, savepath):
    indices_covered = set()

    def job_generator(path, files):
        for i in files:
            with open(path + '/' + i, 'r') as f:
                rd = f.read()
            bid = rd.split('\n')[0]
            rd = rd.split('\n')[1:]
            rd = [x.strip().split('\t') for x in rd if len(x) > 0]
            rd = np.array(rd, dtype=np.uint32)
            yield rd, bid

    f = open(path + '/c0_0', 'r')
    temp = f.read()
    f.close()
    ny, nx, z, z_total = [int(x) for x in temp.split('\n')[0].split(':')[2:]]
    piece_size = int(len(temp.split('\n')[1].strip().split('\t')))
    del temp
    z_bins = [[] for i in range(z_total)]
    files = [x for x in os.listdir(path) if re.search(r"^c[0-9]+[_][0-9]+$", x)]
    for i in files:
        f = open(path + '/' + i, 'r')
        temp = f.readline()
        print(temp.split(':'))
        f.close()
        z_bins[int(temp.split(':')[4])].append(i)
    hdf5_file = h5py.File(savepath + '/collated.hdf5', 'w')
    hdf5_file.create_group("collated_uncolored")
    main_group = hdf5_file["collated_uncolored"]
    for i in range(z_total):
        main_group.create_dataset(name=str(i), shape=(nx * piece_size, ny * piece_size), dtype=np.int32,
                                  compression='gzip', compression_opts=9)
        gen = job_generator(path, z_bins[i])
        for piece, bid in gen:
            assert not (0 in piece), "Zero found in array"
            piece_id = [int(x) for x in bid.split(':')[:2]]
            if bid in indices_covered:
                assert False, 'Repeated bigPictureID'
            indices_covered.add(bid)
            location = piece_id[0] * piece_size, piece_id[1] * piece_size
            main_group[str(i)][location[0]:location[0] + piece_size, location[1]:location[1] + piece_size] = piece
            del piece
    hdf5_file.close()


def piece_together_c_large(path):
    indices_covered = set()

    def job_generator(path):
        files = [x for x in os.listdir(path) if re.search(r"^c[0-9]+[_][0-9]+$", x)]
        for i in files:
            with open(path + '/' + i, 'r') as f:
                rd = f.read()
            bid = rd.split('\n')[0]
            rd = rd.split('\n')[1:]
            rd = [x.strip().split('\t') for x in rd if len(x) > 0]
            rd = np.array(rd, dtype=np.uint32)
            yield rd, bid

    f = open(path + '/c0_0', 'r')
    temp = f.read()
    f.close()
    ny, nx = temp.split('\n')[0].split(':')[2:]
    piece_size = int(len(temp.split('\n')[1].strip().split('\t')))
    del temp
    gen = job_generator(path)
    main_array = np.zeros((piece_size * int(ny), piece_size * int(nx)), dtype=np.uint32)
    for piece, bid in gen:
        assert not (0 in piece), "Zero found in array"
        piece_id = [int(x) for x in bid.split(':')[:2]]
        if bid in indices_covered:
            assert False, 'Repeated bigPictureID'
        indices_covered.add(bid)
        location = piece_id[0] * piece_size, piece_id[1] * piece_size
        main_array[location[0]:location[0] + piece_size, location[1]:location[1] + piece_size] = piece
        del piece
    return main_array


def colorizer_linear_capping(m, head_cut, tail_cut, start_color, end_color, img_path):
    ordered_m = np.sort(m.flatten())
    ln = len(ordered_m)
    cutoff_low = ordered_m[int(head_cut / 100 * ln)]
    cutoff_high = ordered_m[-int(tail_cut / 100 * ln)]
    del ordered_m
    m[m < cutoff_low] = cutoff_low
    m[m > cutoff_high] = cutoff_high
    # TODO: currently the color_scale will be too large for small head/tail cuts
    # TODO: In general, optimize space (relevant for super large objects)
    color_scale = list(Color(start_color).range_to(Color(end_color), cutoff_high - cutoff_low + 1))
    color_scale = [(x.red * 255, x.green * 255, x.blue * 255) for x in color_scale]
    colored_m = np.array(color_scale, dtype=np.uint8)[m - cutoff_low]
    im = Image.fromarray(colored_m, 'RGB')
    im.save(img_path)


def colorizer_linear_capping_checkpoints(m, color_checkpoints, img_path):
    # percentages here mean percentage of the whole range
    ln = np.max(m.flatten()) - np.min(m.flatten()) + 1
    color_scale = []
    prev_percentage, prev_color = color_checkpoints[0]
    assert prev_percentage == 0, 'Start the checkpoint list at 0'
    for percentage, color in color_checkpoints[1:]:
        # TODO: Avoid rounding problems causing a mismatched size better.
        # TODO: Horribly space inefficient. Improve by using histograms.
        size = int(np.ceil((percentage - prev_percentage) / 100 * ln))  # ceiling to prevent rounding down
        start_col = Color(prev_color)
        end_col = Color(color)
        reds = np.linspace(start_col.red, end_col.red, size)
        greens = np.linspace(start_col.green, end_col.green, size)
        blues = np.linspace(start_col.blue, end_col.blue, size)
        color_scale += [(int(reds[i] * 255), int(greens[i] * 255), int(blues[i] * 255)) for i in range(len(reds))]
        prev_color = color
        prev_percentage = percentage
    assert percentage == 100, 'Last checkpoint should end at 100'
    colored_m = np.array(color_scale, dtype=np.uint8)[m - np.min(m.flatten())]
    im = Image.fromarray(colored_m, 'RGB')
    im.save(img_path)


def colorizer_linear_capping_checkpoints_2a(m, color_checkpoints, img_path):
    # percentages here mean percentage (approximate) of the data points
    ordered_m = np.sort(m.flatten())
    ln = len(ordered_m)
    color_scale = []
    percentage, prev_color = color_checkpoints[0]
    assert percentage == 0, 'Start the checkpoint list at 0'
    prev_cutoff = 0
    for percentage, color in color_checkpoints[1:]:
        # TODO: Avoid step-up/down problems causing a mismatched size better.
        # TODO: Horribly space inefficient. Improve by using histograms.
        cutoff = ordered_m[int(percentage * (ln - 1) / 100)]
        # TODO: Use 1 higher and 1 lower cutoff based on which is the closest or use a better algorithm altogether
        if cutoff <= prev_cutoff:
            print('Cutoff Overlap: ' + str(cutoff) + ' -> ' + str(prev_cutoff + 1))
            cutoff = prev_cutoff + 1
        start_col = Color(prev_color)
        end_col = Color(color)
        reds = np.linspace(start_col.red, end_col.red, cutoff - prev_cutoff)
        greens = np.linspace(start_col.green, end_col.green, cutoff - prev_cutoff)
        blues = np.linspace(start_col.blue, end_col.blue, cutoff - prev_cutoff)
        color_scale += [(int(reds[i] * 255), int(greens[i] * 255), int(blues[i] * 255)) for i in range(len(reds))]
        prev_color = color
        prev_cutoff = cutoff
    assert percentage == 100, 'Last checkpoint should end at 100'
    colored_m = np.array(color_scale, dtype=np.uint8)[m - np.min(m.flatten())]
    im = Image.fromarray(colored_m, 'RGB')
    im.save(img_path)


def collist_display(collist):
    color_scale = []
    prev_percentage, prev_color = collist[0]
    assert prev_percentage == 0, 'Start the checkpoint list at 0'
    for percentage, color in collist[1:]:
        start_col = Color(prev_color)
        end_col = Color(color)
        reds = np.linspace(start_col.red, end_col.red, (percentage - prev_percentage) * 5)
        greens = np.linspace(start_col.green, end_col.green, (percentage - prev_percentage) * 5)
        blues = np.linspace(start_col.blue, end_col.blue, (percentage - prev_percentage) * 5)
        color_scale += [(int(reds[i] * 255), int(greens[i] * 255), int(blues[i] * 255)) for i in range(len(reds))]
        prev_color = color
        prev_percentage = percentage
    assert percentage == 100, 'Last checkpoint should end at 100'
    x = np.zeros((len(color_scale) * 50, 200, 3), dtype=np.uint8)
    for i in range(len(color_scale)):
        x[i * 50:i * 50 + 50, :, :] = color_scale[i]
    im = Image.fromarray(x, 'RGB')
    im.show()
