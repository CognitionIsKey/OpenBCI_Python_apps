import numpy as np
import pandas as pd
import time



''' OpenBCI Ganglion Simulator '''
''' EXAMPLE:
def handle_sample(sample):
    print(sample.channel_data)
'''
'''
board_sim = OpenBCISimulator("/home/.../.../SUBJ1/SSVEP_8Hz_Trial3_SUBJ1.csv", [list with max 4 channels])

board_sim.start_streaming(handle_sample)
'''


class OpenBCISimulator(object):
    def __init__(self, path, channels, sample_rate=256):
        self.sample_rate = sample_rate
        self.channels = channels
        self.path = pd.read_csv(str(path),
                                engine='python')
        self.path = self.path.iloc[:, self.channels]
        #self.filtered()

    def filtered(self):
        '''Not working at this time.'''
        # TODO: Implement filtering.
        for i in range(len(self.channels)):
            filtered = sig.butter(self.path.iloc[:, i], 256, 49, 51)
            filtered = sig.butter(filtered, 256, 3, 50)
            self.path.iloc[:, i] = filtered
        return self.path

    def start_streaming(self, callback):
        __temp = []
        for i in range(len(self.path)):
            for j in self.path.ix[i]:
                __temp.append(j)
            sample = OpenBCISample(i, __temp)
            __temp = []
            time.sleep(1./self.sample_rate)
            callback(sample)


class OpenBCISample(object):
    """Object encapulsating a single sample from the OpenBCI board."""
    def __init__(self, packet_id, channel_data):
        self.id = packet_id
        self.channel_data = channel_data
