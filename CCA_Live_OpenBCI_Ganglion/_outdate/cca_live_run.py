#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/.../OpenBCI_Python")
import numpy as np
import multiprocessing as mp
import open_bci_ganglion as bci
from sklearn.cross_decomposition import CCA


'''EXAMPLE
test = cca_live()

# up to 12 #
test.add_stimuli(hz1)
test.add_stimuli(hz2)

test.print_results()
'''


class cca_live(object):
    """Online data parser for 4 channels."""
    def __init__(self, sampling_rate=200, connect=True):
        self.sampling_rate = sampling_rate
        self.connect = connect
        self.ref_signals = []
        self.corr_values = []
        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)
        if connect:
            self.init_start()

    def init_start(self):
        self.queue = mp.Queue()
        self.streaming = mp.Event()
        self.terminate = mp.Event()
        self.prcs = mp.Process(target=self.split)
        self.prcs.daemon = True
        self.prcs.start()

    def add_stimuli(self, hz):
        '''Add stimuli to generate artificial signal'''
        self.hz = hz
        self.ref_signals.append(SignalReference(self.hz, self.__t))

    def print_results(self):
        ''' Prints results in terminal '''
        i = 0  # increment for each sample
        while True:
            self.push_sample(self.queue.get())
            print(chr(27) + "[2J")
            print("===========================")
            print("Korelacje dla kanałów:")
            print('Reference HZ : %s' % self.corr_values[i].reference[0].hz)
            print('Channel 1 : %s' % self.corr_values[i].channels[0][0])
            print('Channel 2 : %s' % self.corr_values[i].channels[0][1])
            print('Channel 3 : %s' % self.corr_values[i].channels[0][2])
            print('Channel 4 : %s' % self.corr_values[i].channels[0][3])
            print("===========================")
            i += 1

    def split(self):
        self.increment = 0
        self.__pre_buffer = []

        def handle_sample(sample):
            ''' Save samples into table; single process '''
            __sample_chunk = [sample.channel_data[0],
                              sample.channel_data[1],
                              sample.channel_data[2],
                              sample.channel_data[3]]
            self.__pre_buffer.append(__sample_chunk)

            if len(self.__pre_buffer) == self.sampling_rate + 1:
                del self.__pre_buffer[:1]

            # Event listener #
            if self.board.streaming:
                self.streaming.set()

            if self.terminate.is_set():
                self.streaming.clear()
                self.board.stop()

            self.increment += 1

            # Push #
            if self.increment % 200 == 0:
                self.queue.put(self.__pre_buffer)
                self.increment = 0

        # Board connection #
        self.board = bci.OpenBCIBoard(port="d2:b4:11:81:48:ad")
        self.board.start_streaming(handle_sample)

    def push_sample(self, packet):
        ''' Push single sample into the list '''
        self.corr_values.append(CrossCorrelation(packet,
                                                 self.ref_signals,
                                                 self.__t))


class SignalReference(object):
    ''' Reference signal generator'''
    def __init__(self, hz, t):
        self.hz = hz

        self.sin = np.array([np.sin(2*np.pi*i*self.hz) for i in t])
        self.cos = np.array([np.cos(2*np.pi*i*self.hz) for i in t])

        # TODO: Implement harmonic signals. #
        self.sin_2 = np.array([np.sin(2*np.pi*i*self.hz*2) for i in t])
        self.cos_2 = np.array([np.cos(2*np.pi*i*self.hz*2) for i in t])


class CrossCorrelation(object):
    ''' CCA class; returns correlation value for each channel '''
    packet_id = 0  # compensate for array lenght

    def __init__(self, signal_sample, ref_signals, t):
        self.id = CrossCorrelation.packet_id
        self.signal = np.squeeze(np.array(signal_sample))
        self.reference = ref_signals
        self.channels = np.zeros(shape=(len(self.reference), 4))
        CrossCorrelation.packet_id += 1
        # Check if table not empty #
        try:
            self.correlate(self.signal)
        except:
            print("Error, couldn't find signal to correlate!")

    def correlate(self, signal):
        for ref in range(len(self.reference)):
            for i in range(4):
                sample = np.array([signal[:, i]]).T

                cca1 = CCA(n_components=1)
                cca2 = CCA(n_components=1)

                ref_sin = self.reference[ref].sin
                ref_cos = self.reference[ref].cos

                cca1.fit(sample, ref_sin)
                cca2.fit(sample, ref_cos)

                U, V = cca1.transform(sample, ref_sin)
                U2, V2 = cca2.transform(sample, ref_cos)

                corr = np.corrcoef(U.T, V)[0, 1]
                corr2 = np.corrcoef(U2.T, V2)[0, 1]
                cor = np.round(max(corr, corr2), 4)

                self.channels[ref][i] = abs(cor)


if __name__ == "__main__":
    test = cca_live()
    test.add_stimuli(hz)
    test.print_results()
