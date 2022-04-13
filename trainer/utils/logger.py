import time
import logging


class Logger(object):
    def __init__(self, log_file, log_per_nframes, tags, loss_per_frame=1.0):
        self.log_file = log_file
        self.num_frames = 0
        self.total_frames = 0
        self.loss = [0.0 for x in tags]
        self.total_loss = [0.0 for x in tags]
        self.log_per_nframes = log_per_nframes
        self.tags = tags
        self.loss_per_frame = loss_per_frame
        self.start_time = time.time()
        self.log_time = time.time()

    def update_and_log(self, num_frames, loss):
        self.num_frames += num_frames
        self.total_frames += num_frames
        for i, l in enumerate(loss):
            self.loss[i] += l
            self.total_loss[i] += l
        if self.num_frames >= self.log_per_nframes:
            elapsed = time.time() - self.log_time
            for i, l in enumerate(self.loss):
                self.log_file.write('{}: {:.3f} \t'.format(self.tags[i],
                                    l / self.loss_per_frame / \
                                    float(self.num_frames)))
            self.log_file.write(
                    'fps: {:.3f} k\n'.format(self.num_frames/elapsed/1000))
            self.log_file.flush()
            #reset
            self.num_frames = 0
            self.loss = [0.0 for x in self.tags]
            self.log_time = time.time()

    def summarize_and_log(self):
        for i, l in enumerate(self.total_loss):
            self.log_file.write('Finished, Overall Avg {}:'
                                ' {:.3f}\t'.format(self.tags[i],
                                l / self.loss_per_frame / \
                                float(self.total_frames)))
        elapsed = time.time() - self.start_time
        self.log_file.write('Avg fps:' \
                '{:.3f} k\n'.format(self.total_frames/elapsed/1000))
        return self.total_loss[0], self.total_frames


def set_logger(logger_name, file_name):
    """
    Create logger to log on both stdout and specific file
    """
    logger = logging.getLogger(logger_name)
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=file_name)

    logger_level = logging.DEBUG
    logger.setLevel(logger_level)
    consoleHandler.setLevel(logger_level)
    fileHandler.setLevel(logger_level)

    format_str = "%(asctime)s %(name)s:%(levelname)s %(message)s"
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format_str, time_format)
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger
