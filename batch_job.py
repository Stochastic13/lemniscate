import numpy as np
import decimal
import time
import multiprocessing
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import warnings
import tqdm


# holds the information about a job
class JobPiece(object):

    def __init__(self, prec, max_depth, bounding_box, cutoff, big_picture_id, max_piece):
        self.prec = prec
        self.max_iter = max_depth
        self.region = bounding_box
        self.cutoff = cutoff
        self.result = None
        self.std_streams = None
        self.big_picture_id = big_picture_id
        self.piece_size = max_piece


# The main iterator function which feeds the data to the C subprocess
def iterator_c_subp(job, nid, executable_name, path):
    args_list = [executable_name, "sp", str(job.prec), str(job.region), str(job.max_iter), str(job.cutoff),
                 f'{path}/c{nid}', job.big_picture_id, str(job.region[0][0]), str(job.region[0][1]),
                 str(job.region[1][0]), str(job.region[1][1])]
    s = subprocess.run(args_list, text=True, capture_output=True)
    job.result = s.returncode
    job.std_streams = (s.stdout, s.stderr)
    return job


# Single mandelbrot image from width * height by dividing the job into pieces
def job_creator_large(bounding_box, max_depth, width, height, max_piece, prec, cutoff):
    # max_piece must be an multiple of the width AND height
    decimal.getcontext().prec = prec
    # since the MPFR prec is in bits, this ensures the decimal module prec is more than enough
    n_pieces_x = width // max_piece
    n_pieces_y = height // max_piece
    # calculate the jump between successive pieces
    yjump = decimal.Decimal((bounding_box[1][1] - bounding_box[0][1]) / n_pieces_y)
    xjump = decimal.Decimal((bounding_box[1][0] - bounding_box[0][0]) / n_pieces_x)
    curr_x = decimal.Decimal(bounding_box[0][0])
    index = [0, 0]
    pending = n_pieces_x * n_pieces_y
    while curr_x < bounding_box[1][0]:
        curr_y = decimal.Decimal(bounding_box[0][1])
        while curr_y < bounding_box[1][1]:
            piece_id = f'{index[0]}:{index[1]}:{n_pieces_y}:{n_pieces_x}'
            new_region = ((curr_y, curr_x), (curr_y + yjump, curr_x + xjump))
            sub_job = JobPiece(prec, max_depth, new_region, cutoff, piece_id, max_piece)
            pending -= 1
            yield sub_job, pending
            curr_y += yjump
            index[0] += 1
            index[0] %= n_pieces_y
        curr_x += xjump
        index[1] += 1


class LogFile(object):

    def __init__(self, filename):
        self.f = filename
        self.init_time = time.time()

    def write(self, arg1, arg2):
        with open(self.f, 'a') as file_handle:
            time_to_put = round(time.time() - self.init_time, 4)
            file_handle.write(f'{time_to_put} : {arg2} : {arg1}\n')


def worker_foo_verbose(q_in, q_out, pid, bin_path, path, logfile):
    logfile.write('BORN.', str(pid))
    import queue
    import time
    time.sleep(10)
    big_ind = 0
    while not q_in.empty():
        try:
            job = q_in.get(timeout=10)
            logfile.write('RECEIVED JOB.', str(pid))
            time_s = time.time()
            result = iterator_c_subp(job, f'{pid}_{big_ind}', bin_path, f'{path}/temp_data')
            time_e = time.time()
            big_ind += 1
            logfile.write('FINISHED JOB IN ' + str(round(time_e - time_s, 2)) + ' SECONDS.', str(pid))
            q_out.put(result)
        except queue.Empty:
            logfile.write('JOB PROMISED BUT RECEIVED NONE.', str(pid))
    time.sleep(5)
    logfile.write('DYING.', str(pid))


def worker_main(gen_args, work_q, result_q, death_event, path, logfile):
    logfile.write('BORN.', -1)
    import pickle
    import time
    import queue
    start_time = time.time()
    index = 0
    logfile.write('MAKING GEN.', -1)
    gen = job_creator_large(*gen_args)
    total_tasks = 0
    # TODO: Add support for lazily populated queue
    for i in gen:
        work_q.put(i[0])
        total_tasks += 1
    logfile.write('POPULATED QUEUE.', -1)
    tqdm_obj = tqdm.tqdm(desc='Computation Progress', total=total_tasks, unit='sub-units')
    while not death_event.is_set():
        try:
            result = result_q.get(timeout=10)
            logfile.write('RECEIVED A RESULT.', -1)
            if result.result > 0:
                f = open(f'{path}/errors/{index}', 'wb')
                f.write(pickle.dumps(result))
                f.close()
                warnings.warn(f'Non-Zero Exit code encountered: {result.result}', RuntimeWarning)
                logfile.write(f'BAD RESULT WITH RETURNCODE {result.result}. PICKLE-DUMPED THE JOB DETAILS', -1)
            index += 1
            tqdm_obj.update(1)
        except queue.Empty:
            logfile.write('NO RESULTS.', -1)
        if index == total_tasks:
            death_event.set()
            tqdm_obj.close()
            break
    time.sleep(10)  # wait for all processes to die properly
    logfile.write('DYING.', -1)


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


def run2_c(n_processes=None, max_iter=100, bb=None, cutoff=2, path='.', w=100, h=100, piece=10, prec=32,
           bin_path='./bin/mandelbrot', **kwargs):
    if n_processes is None:
        n_processes = os.cpu_count()
    if bb is None:
        bounding_box = ((decimal.Decimal('-1'), decimal.Decimal('-1')),
                        (decimal.Decimal('1'), decimal.Decimal('1')))
    else:
        bb = bb.split(',')
        bounding_box = ((decimal.Decimal(bb[0]), decimal.Decimal(bb[1])),
                        (decimal.Decimal(bb[2]), decimal.Decimal(bb[3])))
    if not os.path.isdir(f'{path}/errors'):
        os.mkdir(f'{path}/errors')
    if not os.path.isdir(f'{path}/temp_data'):
        os.mkdir(f'{path}/temp_data')
    work_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    gen_args = (bounding_box, max_iter, w, h, piece, prec, cutoff)
    death_event = multiprocessing.Event()
    death_event.clear()
    logfile = LogFile(f'{path}/main_log_{int(time.time())}.log')
    a = (gen_args, work_q, result_q, death_event, path, logfile)
    p1 = multiprocessing.Process(target=worker_main, daemon=True, args=a)
    p1.start()
    print('Started the main_worker process.')
    workers = []
    for pid in range(n_processes):
        a = (work_q, result_q, pid, bin_path, path, logfile)
        p2 = multiprocessing.Process(target=worker_foo_verbose, daemon=True, args=a)
        p2.start()
        workers.append(p2)
    with open(f'{path}/run_config.log', 'w') as f:
        f.write(f'n_processes : {n_processes}\n')
        f.write(f'Width/Height/Piece : {w}/{h}/{piece}\n')
        f.write('Main bounding_box : ' + ','.join([str(x[0]) + '/' + str(x[1]) for x in bounding_box]) + '\n')
        f.write('Computing_function : run2_c\n')
        f.write(f'Starting time : {time.time()}\n')
        death_event.wait()
        p1.join()
        for p in workers:
            p.join()
        f.write(f'Ending Time: {time.time()}')
    print('Successfully Completed!')


if __name__ == '__main__':
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        run2_c(**parse_job_file(filename))
    else:
        run2_c()
