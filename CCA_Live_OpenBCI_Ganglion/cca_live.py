#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/oskar/kopia github/OpenBCI_Python")
import numpy as np
import multiprocessing as mp
import open_bci_ganglion as bci
from sklearn.cross_decomposition import CCA


'''EXAMPLE
test = cca_live()

#Curently works with just one stimuli.
test.add_stimuli(hz)

test.print_results()
'''


class cca_live(object):
    """Online data parser for 4 channels."""
    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        self.ref_signals = []
        self.corr_values = np.zeros(shape=(120, 12), dtype=object)
        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)
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
            self.corr_values[:][i]
            print("Korelacje dla kanałów.")
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

    def push_sample(self, queue):
        ''' Push single sample into the list '''
        single_packet = queue
        for i in range(len(self.ref_signals)):
            self.corr_values[CrossCorrelation.number][i] = CrossCorrelation(
                                                           single_packet,
                                                           self.ref_signals[i],
                                                           self.__t)


class SignalReference(object):
    ''' Reference signal generator'''
    ref_number = 0

    def __init__(self, hz, t):
        self.id = self.ref_number
        SignalReference.ref_number += 1

        self.hz = hz

        self.sin = np.array([np.sin(2*np.pi*i*self.hz) for i in t])
        self.cos = np.array([np.cos(2*np.pi*i*self.hz) for i in t])

        # TODO: Implement harmonic signals. #
        self.sin_2 = np.array([np.sin(2*np.pi*i*self.hz*2) for i in t])
        self.cos_2 = np.array([np.cos(2*np.pi*i*self.hz*2) for i in t])


class CrossCorrelation(object):
    ''' CCA class; returns correlation value for each channel '''
    number = -1  # compensate for array lenght

    def __init__(self, signal_sample, ref_signals, t):
        self.signal = np.squeeze(np.array(signal_sample))
        self.reference = ref_signals
        self.__channels = [0, 0, 0, 0]
        CrossCorrelation.number += 1
        # Check if table not empty #
        if len(self.signal) <= 1:
            pass
        else:
            self.correlate(self.signal)

        return self.print_channels

    def correlate(self, signal):
        for i in range(len(self.__channels)):
            sample = np.array([signal[:, i]]).T

            cca1 = CCA(n_components=1)
            cca2 = CCA(n_components=1)

            ref_sin = self.reference.sin
            ref_cos = self.reference.cos

            cca1.fit(sample, ref_sin)
            cca2.fit(sample, ref_cos)

            U, V = cca1.transform(sample, ref_sin)
            U2, V2 = cca2.transform(sample, ref_cos)

            corr = np.corrcoef(U.T, V)[0, 1]
            corr2 = np.corrcoef(U2.T, V2)[0, 1]
            cor = np.round(max(corr, corr2), 4)

            self.__channels[i] = abs(cor)

    @property
    def print_channels(self):
        print("Reference signal: %s hz." % self.reference.hz)
        print("Channel 1.", self.__channels[0])
        print("Channel 2.", self.__channels[1])
        print("Channel 3.", self.__channels[2])
        print("Channel 4.", self.__channels[3])

    @property
    def channels(self):
        return self.__channels,  self.reference.id
