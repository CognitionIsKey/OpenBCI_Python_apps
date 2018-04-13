#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import OpenBCI_Simulator.OpenBCI_Simulator as bci_sim
import open_bci_ganglion as bci
from sklearn.cross_decomposition import CCA
import scipy.signal as sig
import time


'''EXAMPLE
    test = CcaLive(simulation=True)
    # Example stimuli.
    test.add_stimuli(22)
    test.add_stimuli(8)
    test.add_stimuli(14)
    test.decission()
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

        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)

        self.streaming = mp.Event()
        self.terminate = mp.Event()

    def initialize(self):
        self.prcs = mp.Process(target=self.split, args=(self.ref_signals,))
        self.prcs.daemon = True
        self.prcs.start()

    def add_stimuli(self, hz):
        '''Add stimuli to generate artificial signal'''
        self.hz = hz
        self.ref_signals.append(SignalReference(self.hz, self.__t))

    def decission(self):
        if self.connect:
            self.initialize()

        # TODO: Listner for button.
        # Terminate condition.

        if self.terminate.is_set():
            self.prcs.terminate()
            self.terminate.clear()

    def split(self, ref_signals):
        self.increment = 0
        self.__pre_buffer = []
        self.ref = ref_signals

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
                packet = self.filtering(self.__pre_buffer)
                self.cross = CrossCorrelation(packet, self.ref)
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
        self.ssvep_display = np.zeros(shape=(len(self.rs), 1), dtype=int)

        # Check if table not empty #

        try:
            self.correlate(self.signal)
            self.make_decission()
            self.print_results()
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
        '''Simple SSVEP Classifier'''
        # TODO: Improve ERP detection.
        best_ = 0
        it_ = 0
        for ref in range(len(self.rs)):
            thereshold_01 = self.channels[ref][0] >= 0.375
            thereshold_02 = self.channels[ref][1] >= 0.22
            if thereshold_01 and thereshold_02:
                if self.channels[ref][0] >= best_:
                    best_ = self.channels[ref][0]
                    it_ = ref
                    self.ssvep_display[it_] = 1
                else:
                    self.ssvep_display[ref] = 0
            else:
                self.ssvep_display[ref] = 0

    def print_results(self):
        ''' Prints results in terminal '''
        print("================================")
        print("Packet ID : %s " % self.id)
        print("================================")
        print("Canonical Correlation:")
        for i in range(len(self.rs)):
            print("Signal {0}: {1}".format(str(self.rs[i].hz) + " hz",
                  self.channels[i][0]))
        print("Stimuli detection: {0}".format([str(self.ssvep_display[i])
              for i in range(len(self.ssvep_display))]))


if __name__ == "__main__":
    test = CcaLive(simulation=True)
    # Example stimuli.
    test.add_stimuli(22)
    test.add_stimuli(8)
    test.add_stimuli(14)
    test.decission()
    time.sleep(15)
    # Make sure it's dead.
    if test.prcs.is_alive():
        test.prcs.terminate()
