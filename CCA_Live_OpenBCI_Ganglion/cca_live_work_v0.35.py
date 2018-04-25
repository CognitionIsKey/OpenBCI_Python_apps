#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import multiprocessing as mp
import OpenBCI_Simulator as bci_sim
sys.path.insert(0,'/home/oskar/kopia github/OpenBCI_Python')
import open_bci_ganglion as bci

from sklearn.cross_decomposition import CCA
import scipy.signal as sig
import time

'''EXAMPLE
test = cca_live()

# up to 12 #
test.add_stimuli(hz1)
test.add_stimuli(hz2)

test.print_results()
'''

CHANNELS = [14, 15, 22, 27]
BCI_PORT = "d2:b4:11:81:48:ad"
PATH = "/home/oskar/Downloads/Baka/SUBJ1/SSVEP_8Hz_Trial1_SUBJ1.csv"
# TODO: Path generator


class CcaLive(object):
    """Online data parser for 4 channels."""
    '''Sampling rate - default 256 samples per second
    connect - start on init.
    simulation - connect to real BCI or use openBCI simulator,
    with your own eeg_file.'''
    def __init__(self, sampling_rate=256, connect=True, simulation=False,
                 port=BCI_PORT):

        # Device parameters
        self.bci_port = port
        self.sampling_rate = sampling_rate
        self.connect = connect
        self.simulation = simulation

        self.reference_signals = []

        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)

        self.streaming = mp.Event()
        self.terminate = mp.Event()

    def initialize(self):
        self.prcs = mp.Process(target=self.split,
                               args=(self.reference_signals,))
        self.prcs.daemon = True
        self.prcs.start()

    def add_stimuli(self, hz):
        '''Add stimuli to generate artificial signal'''
        # TODO: Stimuli support for pygame.
        self.hz = hz
        self.reference_signals.append(SignalReference(self.hz, self.__t))

    def decission(self):
        if self.connect:
            self.initialize()

        # TODO: Listner for button.
        # Terminate condition.
        time.sleep(30)

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

            self.correlation.acquire_data(__sample_chunk)

            # Set termination
            if self.terminate.is_set():
                self.streaming.clear()
                self.board.stop()

        # Board connection #

        if self.simulation:
            self.correlation = CrossCorrelation(self.sampling_rate,
                                                len(CHANNELS),
                                                self.ref)
            self.board = bci_sim.OpenBCISimulator(PATH, CHANNELS)
            self.board.start_streaming(handle_sample)
        else:
            self.correlation = CrossCorrelation(self.sampling_rate,
                                                4,
                                                self.ref)
            self.board = bci.OpenBCIBoard(port=self.bci_port)
            self.board.start_streaming(handle_sample)
            self.board.disconnect()


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
    def __init__(self, sampling_rate, channels_num, ref_signals):
        self.packet_id = 0
        self.all_packet = 0
        self.sampling_rate = sampling_rate
        self.rs = ref_signals
        self.sampling_rate = sampling_rate
        self.signal_window = np.zeros(shape=(sampling_rate, channels_num))
        self.channels = np.zeros(shape=(len(self.rs), 3), dtype=tuple)
        self.ssvep_display = np.zeros(shape=(len(self.rs), 1), dtype=int)

        # Check if table not empty #

        try:
            self.correlate(self.signal_window)
            self.make_decission()
            self.print_results()
        except TypeError as e:
            print(e)

    def acquire_data(self, packet):
        self.signal_window[self.packet_id] = packet
        self.packet_id += 1

        if self.packet_id % self.sampling_rate == 0:
            self.all_packet += 1
            filtered = self.filtering(self.signal_window)
            self.correlate(filtered)
            self.make_decission()
            self.print_results()
            self.packet_id = 0

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
        print("Packet ID : %s " % self.all_packet)
        print("================================")
        print("Canonical Correlation:")
        for i in range(len(self.rs)):
            print("Signal {0}: {1}".format(str(self.rs[i].hz) + " hz",
                  self.channels[i][0]))
        print("Stimuli detection: {0}".format([str(self.ssvep_display[i])
              for i in range(len(self.ssvep_display))]))


if __name__ == "__main__":
    test = CcaLive()
    # Example stimuli.
    test.add_stimuli(22)
    test.add_stimuli(8)
    test.add_stimuli(14)
    test.decission()

    # Make sure it's dead.
    if test.prcs.is_alive():
        print("It was alive!")
        test.prcs.terminate()
