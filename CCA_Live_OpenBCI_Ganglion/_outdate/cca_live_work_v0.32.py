#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import OpenBCI_Simulator as bci_sim
import open_bci_ganglion as bci
from sklearn.cross_decomposition import CCA
import scipy.signal as sig


'''EXAMPLE
test = cca_live()

# up to 12 #
test.add_stimuli(hz1)
test.add_stimuli(hz2)

test.print_results()
'''

HZ = 8

CHANNELS = [15, 22, 29, 39]
BCI_PORT = "d2:b4:11:81:48:ad"
PATH = "/home/oskar/Downloads/SUBJ1/SSVEP_{0}Hz_Trial1_SUBJ1.csv".format(HZ)


class CcaLive(object):
    """Online data parser for 4 channels."""
    '''Sampling rate - default 256 samples per second
    connect - start on init.
    simulation - connect to real BCI or use openBCI simulator,
    with your own eeg_file.'''
    def __init__(self, sampling_rate=256, connect=True, simulation=False,
                 port=BCI_PORT):
        self.BCI_PORT = port
        self.sampling_rate = sampling_rate
        self.connect = connect
        self.simulation = simulation
        self.ref_signals = []
        self.corr_values = []
        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)

        self.prcs = mp.Process(target=self.split)
        self.prcs.daemon = True
        self.queue = mp.Queue()
        self.streaming = mp.Event()
        self.terminate = mp.Event()

    def initialize(self):
        self.prcs.start()

    def add_stimuli(self, hz):
        '''Add stimuli to generate artificial signal'''
        self.hz = hz
        self.ref_signals.append(SignalReference(self.hz, self.__t))

    def print_results(self, packet_number):
        ''' Prints results in terminal '''
        i = packet_number
        print(chr(27) + "[2J")
        for j in range(len(self.ref_signals)):
            print("=============================")
            print("Korelacje dla kanałów:")
            print("Packet ID : %s " % self.corr_values[i].id)
            print('Stimuli HZ : %s' % self.corr_values[i].rs[j].hz)
            print('Correlation : %s' % self.corr_values[i].channels[j][0])
            print("========Harmoniczny===========")
            print('Correlation : %s' % self.corr_values[i].channels[j][1])
            print("==========Łączny==============")
            print('Correlation : %s' % self.corr_values[i].channels[j][2])
            print("==============================")
            print("Badany patrzył na bodziec: {0}".format(
                self.corr_values[i].dec_made[j]))

    def decission(self):
        if self.connect:
            self.initialize()
        self.streaming.set()
        packet_number = 0
        while packet_number != 23:
            self.push_sample()
            self.print_results(packet_number)
            packet_number += 1

        self.prcs.terminate()

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
            if self.streaming.is_set():
                if self.increment % self.sampling_rate == 0:
                    packet = self.filtering(self.__pre_buffer)
                    self.queue.put(packet)
                    self.increment = 0

            if self.terminate.is_set():
                self.streaming.clear()
                self.board.stop()

        # Board connection #

        if self.simulation:
            self.board = bci_sim.OpenBCISimulator(PATH, CHANNELS)
            self.board.start_streaming(handle_sample)
        else:
            self.board = bci.OpenBCIBoard(port=self.BCI_PORT)
            self.board.start_streaming(handle_sample)
            self.board.disconnect()

    def filtering(self, packet):
        """ Push single sample into the list """
        packet = np.squeeze(np.array(packet))

        # Butter bandstop filter 49-51hz
        for i in range(4):
            signal = packet[:, i]
            lowcut = 49/(self.sampling_rate*0.5)
            highcut = 51/(self.sampling_rate*0.5)
            [b, a] = sig.butter(4, [lowcut, highcut], 'bandstop')
            packet[:, i] = sig.filtfilt(b, a, signal)

        # Butter bandpass filter 3-49hz
        for i in range(4):
            signal = packet[:, i]
            lowcut = 3/(self.sampling_rate*0.5)
            highcut = 49/(self.sampling_rate*0.5)
            [b, a] = sig.butter(4, [lowcut, highcut], 'bandpass')
            packet[:, i] = sig.filtfilt(b, a, signal)

        return packet

    def push_sample(self):
        if self.streaming.is_set():
            self.corr_values.append(CrossCorrelation(self.queue.get(),
                                                     self.ref_signals))

    def make_stats(self):
        self.correlation_main = []
        self.correlation_harm = []
        self.correlation_all = []
        self.correlation_dec = []

        for i in range(len(self.corr_values)):
            self.correlation_main.append(self.corr_values[i].channels[0][0])
            self.correlation_harm.append(self.corr_values[i].channels[0][1])
            self.correlation_all.append(self.corr_values[i].channels[0][2])
            self.correlation_dec.append(self.corr_values[i].dec_made[0])

class SignalReference(object):
    """ Reference signal generator"""
    def __init__(self, hz, t):
        self.hz = hz

        self.reference = np.zeros(shape=(len(t), 4))

        self.sin = np.array([np.sin(2*np.pi*i*self.hz) for i in t])
        self.cos = np.array([np.cos(2*np.pi*i*self.hz) for i in t])

        self.sin_2 = np.array([np.sin(2*np.pi*i*self.hz*2) for i in t])
        self.cos_2 = np.array([np.cos(2*np.pi*i*self.hz*2) for i in t])

        self.reference[:, 0] = self.sin
        self.reference[:, 1] = self.cos
        self.reference[:, 2] = self.sin_2
        self.reference[:, 3] = self.cos_2


class CrossCorrelation(object):
    """CCA class; returns correlation value for each channel """
    packet_id = 0

    def __init__(self, signal_sample, ref_signals):
        self.id = CrossCorrelation.packet_id
        self.signal = signal_sample
        self.rs = ref_signals
        self.channels = np.zeros(shape=(len(self.rs), 3), dtype=tuple)
        CrossCorrelation.packet_id += 1
        self.dec_made = np.zeros(shape=(len(self.rs), 1), dtype=tuple)
        # Check if table not empty #
        try:
            self.correlate(self.signal)
            self.make_decission()
        except TypeError as e:
            print(e)

    def correlate(self, signal):
        for ref in range(len(self.rs)):
            sample = signal

            cca = CCA(n_components=1)
            cca_ref = CCA(n_components=1)
            cca_all = CCA(n_components=1)

            ref_ = self.rs[ref].reference[:, [0, 1]]
            ref_2 = self.rs[ref].reference[:, [2, 3]]
            ref_all = self.rs[ref].reference[:, [0, 1, 2, 3]]

            cca.fit(sample, ref_)
            cca_ref.fit(sample, ref_2)
            cca_all.fit(sample, ref_all)

            u, v = cca.transform(sample, ref_)
            u_2, v_2 = cca_ref.transform(sample, ref_2)
            u_3, v_3 = cca_all.transform(sample, ref_all)

            corr = np.corrcoef(u.T, v.T)[0, 1]
            corr2 = np.corrcoef(u_2.T, v_2.T)[0, 1]
            corr_all = np.corrcoef(u_3.T, v_3.T)[0, 1]
            self.channels[ref] = corr, corr2, corr_all

    def make_decission(self):
        '''SSVEP Classifier.'''
        # TODO: Improve ERP detection.
        best_ = 0
        it_ = 0
        if len(self.rs) == 1:
            for ref in range(len(self.rs)):
                if self.channels[ref][0] >= 0.375 and self.channels[ref][1] >= 0.22:
                    self.dec_made[ref] = True
                else:
                    self.dec_made[ref] = False
        else:
            for ref in range(len(self.rs)):
                # Step 1
                if self.channels[ref][0] >= 0.375 and self.channels[ref][1] >= 0.22:
                    if self.channels[ref][0] >= best_:
                        best_ = self.channels[ref][0]
                        it_ = ref
                else:
                    self.dec_made[ref] = 0
            self.dec_made[it_] = it_


if __name__ == "__main__":
    test = CcaLive(simulation=True)
    # Example stimuli.
    test.add_stimuli(14)
    test.add_stimuli(8)
    test.decission()
    test.make_stats()
    if test.prcs.is_alive():
        test.prcs.terminate()


# Garbage collector
import gc
collected = gc.collect()
print("Garbage collector: collected %s objects." % collected)

# Plotting after end.
import matplotlib.pyplot as plt
plt.plot(test.correlation_main, label="Main reference signal correlation.")
plt.plot(test.correlation_harm, label="Harmonic signal correlation.")
plt.plot(test.correlation_all, label="Combined correlation. (proper CCA)")
plt.plot(test.correlation_dec, label="Decision")
plt.legend()
plt.show()
