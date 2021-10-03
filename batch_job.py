import numpy as np
import h5py
import ctypes
import decimal
import time
import multiprocessing
import pickle
import matplotlib.pyplot as plt
import sys
import subprocess


# TODO: Fix the y before x structure of the Bounding Box coordinates
class JobPiece(object):

    def __init__(self, prec, max_depth, bounding_box, cutoff, bigpictureID, max_piece):
        self.prec = prec
        self.max_iter = max_depth
        self.region = bounding_box
        self.cutoff = cutoff
        self.result = None
        self.bigPictureID = bigpictureID
        self.piece_size = max_piece


# The main iterator function which feeds the data to the C subprocess
def iterator_c_subp(bb, width, max_iter, cutoff, nid, bigpictureid, prec, executable_name, path):
    args_list = ["./" + executable_name, "sp", str(prec), str(width), str(max_iter), str(cutoff),
                 './' + path + '/c' + str(nid), bigpictureid, str(bb[0][1]), str(bb[0][0]), str(bb[1][1]),
                 str(bb[1][0])]
    try:  # the subprocess will exit with code 0 if successful
        subprocess.run(args_list, check=True, capture_output=True)
    except subprocess.CalledProcessError as x:
        print("\nError: Subprocess returned non-zero exit code = " + str(x.returncode))
        print("\tSTDERR: " + str(x.stderr))
        print("\tSTDOUT: " + str(x.stdout))
        return False
    return True


# Single mandelbrot image from width * height by dividing the job into pieces
def job_creator_large(bounding_box, max_depth, width, height, max_piece, prec, cutoff):
    # max_piece must be an multiple of the width AND height
    decimal.getcontext().prec = prec
    # since the MPFR prec is in bits, this ensures the decimal module prec is more than enough
    # TODO: replace this with a better precision value
    n_pieces_x = width // max_piece
    n_pieces_y = height // max_piece
    # calculate the jump between successive pieces
    yjump = decimal.Decimal((bounding_box[1][0] - bounding_box[0][0]) / n_pieces_y)
    xjump = decimal.Decimal((bounding_box[1][1] - bounding_box[0][1]) / n_pieces_x)
    curr_x = decimal.Decimal(bounding_box[0][1])
    index = [0, 0]
    pending = n_pieces_x * n_pieces_y
    while curr_x < bounding_box[1][1]:
        curr_y = decimal.Decimal(bounding_box[0][0])
        while curr_y < bounding_box[1][0]:
            piece_id = str(index[0]) + ':' + str(index[1]) + ':' + str(n_pieces_y) + ':' + str(n_pieces_x)
            new_region = ((curr_y, curr_x), (curr_y + yjump, curr_x + xjump))
            sub_job = JobPiece(prec, max_depth, new_region, cutoff, piece_id, max_piece)
            pending -= 1
            yield sub_job, pending
            curr_y += yjump
            index[0] += 1
            index[0] %= n_pieces_y
        curr_x += xjump
        index[1] += 1


def job_creator_zoom_c_sans_recycle(center, span_start, span_zoom_factor, n_zooms, max_depth, width, height, max_piece,
                                    prec_range, cutoff):
    # ultra-detailed mandelbrot zoom
    # max_piece must be an multiple of the width AND height
    assert float(span_zoom_factor) < 1
    span_zoom_factor = decimal.Decimal(span_zoom_factor)
    center[0] = decimal.Decimal(center[0])
    center[1] = decimal.Decimal(center[1])
    decimal.getcontext().prec = prec_range[-1]
    n_pieces_x = width // max_piece
    n_pieces_y = height // max_piece
    curr_span = [decimal.Decimal(span_start[0]), decimal.Decimal(span_start[1])]
    curr_prec = int(prec_range[0])
    for i in range(n_zooms):
        yjump = curr_span[1] / n_pieces_y
        xjump = curr_span[0] / n_pieces_x
        curr_x = center[0] - curr_span[0] / 2
        index = [0, 0]
        pending = n_pieces_x * n_pieces_y * (n_zooms - i)
        while curr_x < (center[0] + curr_span[0] / 2):
            curr_y = decimal.Decimal(center[1] - curr_span[1] / 2)
            while curr_y < (center[1] + curr_span[1] / 2):
                piece_id = str(index[0]) + ':' + str(index[1]) + ':' + str(n_pieces_y) + ':'
                piece_id += str(n_pieces_x) + ':' + str(i) + ':' + str(n_zooms)
                new_region = ((curr_y, curr_x), (curr_y + yjump, curr_x + xjump))
                sub_job = JobPiece(curr_prec, max_depth, new_region, cutoff, piece_id, max_piece)
                pending -= 1
                yield sub_job, pending
                curr_y += yjump
                index[0] += 1
                index[0] %= n_pieces_y
            curr_x += xjump
            index[1] += 1
        curr_prec = int(curr_prec + prec_range[1] - prec_range[0])
        curr_span[0] *= span_zoom_factor
        curr_span[1] *= span_zoom_factor


def worker_main(gen, gen_args, work_q, result_q, death_event, path, update_freq, logfile):
    logfile.write('BORN.', -1)
    import pickle
    import time
    import queue
    import decimal
    start_time = time.time()
    prev_loop_time = start_time
    index = 0
    logfile.write('MAKING GEN.', -1)
    gen = gen(*gen_args)
    logfile.write('MADE GEN.', -1)
    total_tasks = 0
    for i in gen:
        work_q.put(i[0])
        total_tasks += 1
    logfile.write('POPULATED QUEUE.', -1)
    while not death_event.is_set():
        try:
            result = result_q.get(timeout=update_freq - (time.time() - prev_loop_time))
            if len(result.result) > 0:  # c_subp returns 0
                f = open(path + '/' + str(index), 'wb')
                f.write(pickle.dumps(result))
                f.close()
            index += 1
            logfile.write('RECEIVED A RESULT.', -1)
        except queue.Empty:
            logfile.write('NO RESULTS.', -1)
        if (time.time() - prev_loop_time) >= update_freq:
            logfile.write('UPDATING STATUS.', -1)
            prev_loop_time = time.time()
            print('Tasks in queue/total tasks: ', work_q.qsize(), '/', total_tasks)
            print('Total results saved: ', index)
            print('Total time elapsed (minutes): ', round((time.time() - start_time) / 60, 2))
            print('-----------------------------')
        if index == total_tasks:
            death_event.set()
            continue
    time.sleep(5)  # wait for all processes to die properly
    logfile.write('DYING.', -1)


class logfile(object):

    def __init__(self, filename):
        self.f = filename
        self.init_time = time.time()

    def write(self, arg1, arg2):
        with open(self.f, 'a') as file_handle:
            file_handle.write(str(round(time.time() - self.init_time, 4)))
            file_handle.write(' : ')
            file_handle.write(str(arg2))
            file_handle.write(' : ')
            file_handle.write(str(arg1))
            file_handle.write('\n')


def worker_foo_verbose(q_in, q_out, foo, foo2, pid, logfile):
    logfile.write('BORN.', str(pid))
    import decimal
    import numpy as np
    import queue
    import time
    time.sleep(10)
    bigind = 0
    while not q_in.empty():
        try:
            job = q_in.get(timeout=10)
            logfile.write('RECEIVED JOB.', str(pid))
            time_s = time.time()
            result = foo(job.region, job.piece_size, job.piece_size, job.max_iter, job.cutoff, foo2,
                         str(pid) + '_' + str(bigind), job.bigPictureID, job.prec)
            time_e = time.time()
            job.update_result(result)
            bigind += 1
            logfile.write('FINISHED JOB IN ' + str(round(time_e - time_s, 2)) + ' SECONDS.', str(pid))
            q_out.put(job)
        except queue.Empty:
            logfile.write('JOB PROMISED BUT RECEIVED NONE.', str(pid))
    time.sleep(5)
    logfile.write('DYING.', str(pid))


def worker_cpu_logger(filename, update_freq, death_event, verbosity=1, cpu_usage_interval=1):
    assert cpu_usage_interval < update_freq, 'Cannot update faster than interval length'
    import psutil
    import time
    start_time = time.time()
    processes_to_log = psutil.Process().parent().children()
    process_cpu_times = []
    for i in processes_to_log:
        cpu_times = i.cpu_times()
        process_cpu_times.append((cpu_times.user, cpu_times.system))
    process_cpu_times_start = [str(x[0]) + ',' + str(x[1]) for x in process_cpu_times]
    if verbosity == 2:  # only in Linux
        sensor_names = [x.label for x in psutil.sensors_temperatures()['coretemp']]
        columns = 'time:' + ':'.join(['sysTemp_' + str(i) for i in sensor_names])
        columns += ':' + ':'.join(['coreTime_' + str(i) for i in range(psutil.cpu_count())])

        def log_function():
            temperature = psutil.sensors_temperatures()['coretemp']
            assert [x.label for x in temperature] == sensor_names, 'Sensor_name_mismatch'
            temperature = [str(x.current) for x in temperature]
            general_cpu_times = psutil.cpu_times(percpu=True)
            s2 = ':'.join([str(x.user) + ',' + str(x.system) + ',' + str(x.idle) for x in general_cpu_times])
            return ':'.join(temperature) + ':' + s2
    elif verbosity == 1:
        columns = 'time:' + ':'.join(['coreTime_' + str(i) for i in range(psutil.cpu_count())])

        def log_function():
            general_cpu_times = psutil.cpu_times(percpu=True)
            return ':'.join([str(x.user) + ',' + str(x.system) + ',' + str(x.idle) for x in general_cpu_times])
    elif verbosity == 0:
        columns = 'time:' + ':'.join(['coreUsage_' + str(i) for i in range(psutil.cpu_count())])

        def log_function():
            general_cpu_usage = psutil.cpu_percent(interval=cpu_usage_interval, percpu=True)
            return ':'.join(str(x) for x in general_cpu_usage)
    else:
        assert False, 'Invalid verbosity level'
    with open(filename, 'w') as file:
        file.write(columns + '\n')
    while not death_event.is_set():
        prev_loop_time = time.time()
        with open(filename, 'a') as file:
            file.write(str(round(time.time() - start_time, 4)))
            file.write(':')
            file.write(log_function() + '\n')
            old_times = process_cpu_times[:]  # in case the process died before this logging
            process_cpu_times = []
            index = 0
            for i in processes_to_log:
                if i.is_running():
                    cpu_times = i.cpu_times()
                    process_cpu_times.append((cpu_times.user, cpu_times.system))
                else:
                    process_cpu_times.append(old_times[index])
                index += 1
            # TODO: Currently, the process cpu_times are accurate only up to +- update_freq
            process_cpu_times_end = [str(x[0]) + ',' + str(x[1]) for x in process_cpu_times]
        time.sleep(update_freq - (time.time() - prev_loop_time))  # relevant for when the log_function takes time
    with open(filename, 'a') as file:
        file.write('\nProcess CPU times:\n')
        file.write('Start> ' + ':'.join(process_cpu_times_start) + '\n')
        file.write('End> ' + ':'.join(process_cpu_times_end) + '\n')


def display_stupid_foo(filename):
    f = open(filename, 'rb')
    result = pickle.loads(f.read())
    result.check_result()
    f.close()
    fig, ax = plt.subplots()
    ax.imshow(result.result)
    plt.show()
    fig, ax = plt.subplots()
    result.blowup_stats(ax)
    plt.show()


def parse_job_file(filename):
    with open(filename) as f:
        rd = f.read()
        rd = rd.split('\n')
    rd = [x.split(' : ') for x in rd if len(x) > 0]
    output = {}
    for k, v in rd:
        try:
            output[k] = int(v)
        except ValueError:
            output[k] = v
    return output


def run3_c_zoom(np=5, mi=1000, cntr=None, c=2, uf=10, pth='trial', w=500, h=500, piece=100, precr="64,128",
                spnstrt="2,2",
                spnzm='0.1', nz=10, **kwargs):
    # center, span_start, span_zoom_factor, n_zooms, max_depth, width, height, max_piece,
    #                                     prec_range, cutoff
    n_processes = np
    max_iter = mi
    cutoff = c
    update_freq = uf
    path = pth + '/c_data'
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    precr = [int(x) for x in precr.split(',')]
    gen_args = (cntr.split(','), spnstrt.split(","), spnzm, nz, max_iter, w, h, piece, precr, cutoff)
    death_event = multiprocessing.Event()
    death_event.clear()
    logf = logfile(path + '/c_main_log.log')
    a = (job_creator_zoom_c_sans_recycle, gen_args, work_q, result_q, death_event, path, update_freq, logf)
    p = multiprocessing.Process(target=worker_main, daemon=True, args=a)
    p.start()
    print('Started the main_worker process.')
    for pid in range(n_processes):
        a = (work_q, result_q, iterator_c_subp, iterator_unit, pid, logf)
        p = multiprocessing.Process(target=worker_foo_verbose, daemon=True, args=a)
        p.start()
    with open(path + '/c_config.log', 'w') as f:
        f.write('n_processes : ' + str(n_processes) + '\n')
        f.write('update_freq : ' + str(update_freq) + '\n')
        f.write('Width/Height/Piece : ' + str(w) + '/' + str(h) + '/' + str(piece) + '\n')
        f.write('Center : ' + cntr + '\n')
        f.write('Span Start/Factor : ' + str(spnstrt) + '/' + str(spnzm) + '\n')
        f.write('Computing_function : run3_c_zoom\n')
        f.write('Starting time : ' + str(time.time()) + '\n')
        death_event.wait()
        f.write('Ending Time: ' + str(time.time()))
    print('Successfully Completed!')


def run1(np=5, mi=1000, bb=None, c=2, uf=10, pth='trial', w=500, h=500, piece=100, prec=40, **kwargs):
    n_processes = np
    max_iter = mi
    if bb is None:
        bounding_box = ((decimal.Decimal('-1'), decimal.Decimal('-1')),
                        (decimal.Decimal('1'), decimal.Decimal('1')))
    else:
        bb = bb.split(',')
        bounding_box = ((decimal.Decimal(bb[0]), decimal.Decimal(bb[1])),
                        (decimal.Decimal(bb[2]), decimal.Decimal(bb[3])))
    cutoff = c
    update_freq = uf
    path = pth
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    gen_args = (bounding_box, max_iter, w, h, piece, prec, cutoff)
    death_event = multiprocessing.Event()
    death_event.clear()
    logf = logfile(path + '/main_log.log')
    a = (job_creator_large, gen_args, work_q, result_q, death_event, path, update_freq, logf)
    p = multiprocessing.Process(target=worker_main, daemon=True, args=a)
    p.start()
    print('Started the main_worker process.')
    for pid in range(n_processes):
        a = (work_q, result_q, iterator_region_verbose, iterator_unit, pid, logf)
        p = multiprocessing.Process(target=worker_foo_verbose, daemon=True, args=a)
        p.start()
    with open(path + '/config.log', 'w') as f:
        f.write('n_processes : ' + str(n_processes) + '\n')
        f.write('update_freq : ' + str(update_freq) + '\n')
        f.write('Width/Height/Piece : ' + str(w) + '/' + str(h) + '/' + str(piece) + '\n')
        f.write('Main bounding_box : ' + ','.join([str(x[0]) + '/' + str(x[1]) for x in bounding_box]) + '\n')
        f.write('Computing_function : run1\n')
        f.write('Starting time : ' + str(time.time()) + '\n')
        death_event.wait()
        f.write('Ending Time: ' + str(time.time()))
    print('Successfully Completed!')


def run2_c(np=5, mi=1000, bb=None, c=2, uf=10, pth='trial', w=500, h=500, piece=100, prec=128, **kwargs):
    if bb is None:
        bounding_box = ((decimal.Decimal('-1'), decimal.Decimal('-1')),
                        (decimal.Decimal('1'), decimal.Decimal('1')))
    else:
        bb = bb.split(',')
        bounding_box = ((decimal.Decimal(bb[0]), decimal.Decimal(bb[1])),
                        (decimal.Decimal(bb[2]), decimal.Decimal(bb[3])))
    n_processes = np
    max_iter = mi
    cutoff = c
    update_freq = uf
    path = pth + '/c_data'
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    gen_args = (bounding_box, max_iter, w, h, piece, prec, cutoff)
    death_event = multiprocessing.Event()
    death_event.clear()
    logf = logfile(path + '/c_main_log.log')
    a = (job_creator_large, gen_args, work_q, result_q, death_event, path, update_freq, logf)
    p = multiprocessing.Process(target=worker_main, daemon=True, args=a)
    p.start()
    print('Started the main_worker process.')
    for pid in range(n_processes):
        a = (work_q, result_q, iterator_c_subp, iterator_unit, pid, logf)
        p = multiprocessing.Process(target=worker_foo_verbose, daemon=True, args=a)
        p.start()
    with open(path + '/c_config.log', 'w') as f:
        f.write('n_processes : ' + str(n_processes) + '\n')
        f.write('update_freq : ' + str(update_freq) + '\n')
        f.write('Width/Height/Piece : ' + str(w) + '/' + str(h) + '/' + str(piece) + '\n')
        f.write('Main bounding_box : ' + ','.join([str(x[0]) + '/' + str(x[1]) for x in bounding_box]) + '\n')
        f.write('Computing_function : run2_c\n')
        f.write('Starting time : ' + str(time.time()) + '\n')
        death_event.wait()
        f.write('Ending Time: ' + str(time.time()))
    print('Successfully Completed!')


def run1_cpu(np=5, mi=1000, bb=None, c=2, uf=10, pth='trial', w=500, h=500, piece=100, prec=40, cu=1, v=1, ufc=10,
             **kwargs):
    n_processes = np
    max_iter = mi
    if bb is None:
        bounding_box = ((decimal.Decimal('-1'), decimal.Decimal('-1')),
                        (decimal.Decimal('1'), decimal.Decimal('1')))
    else:
        bb = bb.split(',')
        bounding_box = ((decimal.Decimal(bb[0]), decimal.Decimal(bb[1])),
                        (decimal.Decimal(bb[2]), decimal.Decimal(bb[3])))
    cutoff = c
    update_freq = uf
    update_freq_cpu = ufc
    path = pth
    cpu_usage_interval = cu  # not needed except verbosity == 0
    verbosity_level = v
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    gen_args = (bounding_box, max_iter, w, h, piece, prec, cutoff)
    death_event = multiprocessing.Event()
    death_event.clear()
    logf = logfile(path + '/main_log.log')
    a = (job_creator_large, gen_args, work_q, result_q, death_event, path, update_freq, logf)
    p_main = multiprocessing.Process(target=worker_main, daemon=True, args=a)
    p_main.start()
    print('Started the main_worker process.')
    for pid in range(n_processes):
        a = (work_q, result_q, iterator_region_verbose, iterator_unit, pid, logf)
        p = multiprocessing.Process(target=worker_foo_verbose, daemon=True, args=a)
        p.start()
    with open(path + '/config.log', 'w') as f:
        f.write('n_processes : ' + str(n_processes) + '\n')
        f.write('update_freq : ' + str(update_freq) + '\n')
        f.write('Width/Height/Piece : ' + str(w) + '/' + str(h) + '/' + str(piece) + '\n')
        f.write('Main bounding_box : ' + ','.join([str(x[0]) + '/' + str(x[1]) for x in bounding_box]) + '\n')
        f.write('Computing_function : run1_cpu\n')
        f.write('Starting time : ' + str(time.time()) + '\n')
    a = (path + '/cpu_log.log', update_freq_cpu, death_event, verbosity_level, cpu_usage_interval)
    p_cpu = multiprocessing.Process(target=worker_cpu_logger, args=a, daemon=True)
    p_cpu.start()
    death_event.wait()
    with open(path + '/config.log', 'a') as f:
        f.write('Ending Time: ' + str(time.time()))
    p_main.join()
    p_cpu.join()
    print('Successfully Completed!')


if __name__ == '__main__':
    filename = sys.argv[1]
    run_name = eval(sys.argv[2])
    run_name(**parse_job_file(filename))
