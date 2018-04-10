#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import OpenBCI_Simulator as bci_sim
import open_bci_ganglion as bci
from sklearn.cross_decomposition import CCA



'''EXAMPLE
test = cca_live()

# up to 12 #
test.add_stimuli(hz1)
test.add_stimuli(hz2)

test.print_results()
'''

# If you have eeg with more than 4 channels, please select just 4 for further processing.
CHANNELS = [15, 22, 29, 39]
# OpenBCI MAC adress.
BCI_PORT = "d2:b4:11:81:48:ad"
# Path for eeg_file
PATH = "/home/.../.../SUBJ1/SSVEP_8Hz_Trial2_SUBJ1.csv"


class cca_live(object):
    """Online data parser for 4 channels."""
    '''Sampling rate - default 256 samples per second
    connect - start on init.
    simulation - connect to real BCI or use openBCI simulator,
    with your own eeg_file.'''
    def __init__(self, sampling_rate=256, connect=True, simulation=False):
        self.sampling_rate = sampling_rate
        self.connect = connect
        self.simulation = simulation
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
        ''' channels[x][y][z] stands for x-reference signals
        y-4 elements column with tuples
        z-index of tuple'''
        i = 0  # increment for each sample
        while i != 23:
            self.push_sample(self.queue.get())
            print(chr(27) + "[2J")
            for j in range(len(self.ref_signals)):
                print("===========================")
                print("Korelacje dla kanałów:")
                print("Packet ID : %s " % self.corr_values[i].id)
                print('Stimuli HZ : %s' % self.corr_values[i].reference[j].hz)
                print('Channel 1 : %s' % self.corr_values[i].channels[j][0][0])
                print('Channel 2 : %s' % self.corr_values[i].channels[j][1][0])
                print('Channel 3 : %s' % self.corr_values[i].channels[j][2][0])
                print('Channel 4 : %s' % self.corr_values[i].channels[j][3][0])
                print("========Harmoniczny===========")
                print('Channel 1 : %s' % self.corr_values[i].channels[j][0][1])
                print('Channel 2 : %s' % self.corr_values[i].channels[j][1][1])
                print('Channel 3 : %s' % self.corr_values[i].channels[j][2][1])
                print('Channel 4 : %s' % self.corr_values[i].channels[j][3][1])
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

            self.increment += 1

            # Push #
            if self.increment % self.sampling_rate == 0:
                self.queue.put(self.__pre_buffer)
                self.increment = 0

        # Board connection #
        if self.simulation:
            self.board = bci_sim.OpenBCISimulator(PATH, CHANNELS)
        else:
            self.board = bci.OpenBCIBoard(port=BCI_PORT)

        self.board.start_streaming(handle_sample)

    def push_sample(self, packet):
        ''' Push single sample into the list '''
        packet = np.squeeze(np.array(packet))
        # TODO: Here will be live butter filter.
        # Function to rebuild.

        self.corr_values.append(CrossCorrelation(packet,
                                                 self.ref_signals,
                                                 self.__t))


class SignalReference(object):
    ''' Reference signal generator'''
    def __init__(self, hz, t):
        self.hz = hz

        self.sin = np.array([np.sin(2*np.pi*i*self.hz) for i in t])
        self.cos = np.array([np.cos(2*np.pi*i*self.hz) for i in t])

        self.sin_2 = np.array([np.sin(2*np.pi*i*self.hz*2) for i in t])
        self.cos_2 = np.array([np.cos(2*np.pi*i*self.hz*2) for i in t])


class CrossCorrelation(object):
    ''' CCA class; returns correlation value for each channel '''
    packet_id = 0

    def __init__(self, signal_sample, ref_signals, t):
        self.id = CrossCorrelation.packet_id
        self.signal = signal_sample
        self.reference = ref_signals
        self.channels = np.zeros(shape=(len(self.reference), 4), dtype=tuple)
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
                cca3 = CCA(n_components=1)
                cca4 = CCA(n_components=1)

                ref_sin = self.reference[ref].sin
                ref_cos = self.reference[ref].cos

                ref_sin_2 = self.reference[ref].sin_2
                ref_cos_2 = self.reference[ref].cos_2

                cca1.fit(sample, ref_sin)
                cca2.fit(sample, ref_cos)

                cca3.fit(sample, ref_sin_2)
                cca4.fit(sample, ref_cos_2)

                U, V = cca1.transform(sample, ref_sin)
                U2, V2 = cca2.transform(sample, ref_cos)

                U3, V3 = cca3.transform(sample, ref_sin_2)
                U4, V4 = cca4.transform(sample, ref_cos_2)

                corr = np.corrcoef(U.T, V)[0, 1]
                corr2 = np.corrcoef(U2.T, V2)[0, 1]
                corr3 = np.corrcoef(U3.T, V3)[0, 1]
                corr4 = np.corrcoef(U4.T, V4)[0, 1]

                cor = np.round(max(corr, corr2), 4)
                cor2 = np.round(max(corr3, corr4), 4)

                self.channels[ref][i] = abs(cor), abs(cor2)


if __name__ == "__main__":
    test = cca_live(simulation=True)
    test.add_stimuli(hz1)
    test.add_stimuli(hz2)
    test.add_stimuli(hz...)
    test.print_results()
