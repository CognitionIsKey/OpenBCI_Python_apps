#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/.../OpenBCI_Python/")
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
        self.sig_samples = []
        self.corr_values = []
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
        # TODO: Support for more than 1 stimuli.

        self.hz = hz
        self.ref_signals.append(SignalReference(self.hz, self.__t))
        self.push_sample()

    def print_results(self):
        ''' Prints results in terminal '''
        i = 0  # increment for each sample
        while True:
            self.sig_samples.append(self.queue.get())
            self.push_sample()
            print(chr(27) + "[2J")
            self.corr_values[i].print_channels
            print("Korelacje dla kanałów.")
            i += 1

    def split(self):
        self.__matrix = np.zeros(shape=(200, 4))
        self.increment = -1

        def handle_sample(sample):
            ''' Save samples into table; single process '''
            self.__matrix[self.increment][0] = sample.channel_data[0]
            self.__matrix[self.increment][1] = sample.channel_data[1]
            self.__matrix[self.increment][2] = sample.channel_data[2]
            self.__matrix[self.increment][3] = sample.channel_data[3]

            # Data parser #
            if self.increment == self.sampling_rate - 1:
                self.queue.put(self.__matrix)
                self.__matrix = np.zeros(shape=(200, 4))
                self.increment = -1

            # Event listener #
            if self.board.streaming:
                self.streaming.set()

            if self.terminate.is_set():
                self.streaming.clear()
                self.board.stop()

            self.increment += 1
        # Board connection #
        self.board = bci.OpenBCIBoard(port="d2:b4:11:81:48:ad")
        self.board.start_streaming(handle_sample)

    def push_sample(self):
        ''' Push single sample into the list '''
        # FIXME: Iteration over ref_signals don't work.
        for i in range(len(self.ref_signals)):
            self.corr_values.append((CrossCorrelation(self.sig_samples[-1:],
                                                      self.ref_signals[i],
                                                      i, self.__t)))


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
    def __init__(self, signal_sample, ref_signals, num, t):
        self.signal = np.squeeze(np.array(signal_sample))
        self.reference = ref_signals
        self.__channels = [0, 0, 0, 0]

        # Check if table not empty #
        if len(self.signal) <= 1:
            pass
        else:
            self.correlate(self.signal)

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
        print("Channel 1.", self.__channels[0])
        print("Channel 2.", self.__channels[1])
        print("Channel 3.", self.__channels[2])
        print("Channel 4.", self.__channels[3])

    @property
    def channels(self):
        return self.__channels
