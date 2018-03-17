import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

eeg_file = pd.read_csv("/home/..../")
'''
# EXAMPLE # for offline data
test = cca_live(double=True)
test.split(filtered_data)

test.add_stimuli(hz)
test.optimize_calc()

# Accessing main reference signal #
# plt.plot(test.corr_matrix.T[0],label='Signal 1')
# plt.plot(test.corr_matrix.T[1],label='Signal 2')
# plt.plot(test.corr_matrix.T[2],label='Signal 3')
# Accesing harmonic function for signal #
# plt.plot(test.corr_matrix.T[3],label='Signal 4')
# plt.plot(test.corr_matrix.T[4],label='Signal 5')
# plt.plot(test.corr_matrix.T[5],label='Signal 6')
plt.plot(test.optimized, label="Optimized (main)")
plt.plot(test.optimized_h, label="Optimized (*2)
plt.legend()
plt.show()


'''


class cca_live(object):
    """On irony currently just offline parsing.
  Args:
    sampling_rate: samples per second
    double: makes another reference signal with double of it's value.
  """
    def __init__(self, sampling_rate=256, double=False):
        self.__double = double
        self.sampling_rate = sampling_rate
        self.ref_signals = []
        self.sig_samples = []
        self.corr_values = []
        self.corr_matrix = np.zeros(shape=(0, 0))
        self.__fs = 1./sampling_rate
        self.__t = np.arange(0.0, 1.0, self.__fs)

    def add_stimuli(self, hz):
        '''Add stimuli to generate artificial signal'''
        self.hz = hz
        self.ref_signals.append(SignalReference(self.hz, self.__t))
        self.push_sample()

    def split(self, filtered_data):
        '''Filters data into 1 seconds samples.(256 per sample)'''
        __temp = []
        for i in filtered_data:
            __temp.append(i)
            if len(__temp) == self.sampling_rate:
                sample = np.array(__temp)
                self.sig_samples.append(sample)
                __temp = []
        self.corr_matrix = np.zeros(shape=((len(self.sig_samples)), 6))

    def compare_sample(self):
        '''Function to acquire data - useless for now'''
        def handle_sample(sample):
            self.channel1 = sample.channel_data[0]
            self.channel2 = sample.channel_data[1]
            self.channel3 = sample.channel_data[2]
            self.channel4 = sample.channel_data[3]

    def push_sample(self):
        '''Making objects from samples to easier pick up later
        sig_sample: 256 element array
        ref_signals: reference signals
        j: number of sample
        __double: passing boolean to decide how many signals will be correlate
        '''
        for i in range(len(self.ref_signals)):
            for j in range(len(self.sig_samples)):
                self.corr_values.append((CrossCorrelation(self.sig_samples[j],
                                                          self.ref_signals[i],
                                                          j, self.__double)))
        self.save()

    def save(self):
        '''Saving values to matrix'''
        for i in range(len(self.corr_values)):
            self.corr_matrix[self.corr_values[i].num_sec][0] = self.corr_values[i].score[0]
            self.corr_matrix[self.corr_values[i].num_sec][1] = self.corr_values[i].score[1]
            self.corr_matrix[self.corr_values[i].num_sec][2] = self.corr_values[i].score[2]
            if self.__double:
                self.corr_matrix[self.corr_values[i].num_sec][3] = self.corr_values[i].score_2[0]
                self.corr_matrix[self.corr_values[i].num_sec][4] = self.corr_values[i].score_2[1]
                self.corr_matrix[self.corr_values[i].num_sec][5] = self.corr_values[i].score_2[2]
        self.corr_matrix = abs(self.corr_matrix)

    def optimize_calc(self):
        '''Pick up maximum values in row to choose best signal'''
        self.better_matrix = self.corr_matrix
        self.optimized = []
        self.optimized_h = []
        for i in range(len(self.sig_samples)):
            self.optimized.append(max(self.better_matrix[i][:2]))
        for i in range(len(self.sig_samples)):
            self.optimized_h.append(max(self.better_matrix[i][3:]))


class SignalReference(object):
    ''' Reference signal generator'''
    def __init__(self, hz, t):
        self.hz = hz
        __hz = hz - 0.2
        self.sin_sub = np.array([np.sin(2*np.pi*i*__hz+0.28*np.pi) for i in t])
        self.cos_sub = np.array([np.cos(2*np.pi*i*__hz+0.28*np.pi) for i in t])

        __hz = hz
        self.sin = np.array([np.sin(2*np.pi*i*__hz+0.56*np.pi) for i in t])
        self.cos = np.array([np.cos(2*np.pi*i*__hz+0.56*np.pi) for i in t])

        __hz = hz + 0.2
        self.sin_add = np.array([np.sin(2*np.pi*i*__hz+0.84*np.pi) for i in t])
        self.cos_add = np.array([np.cos(2*np.pi*i*__hz+0.84*np.pi) for i in t])

        __hz = hz - 0.2
        self.sin_sub_2 = np.array([np.sin(2*np.pi*i*__hz+0.28*2*np.pi) for i in t])
        self.cos_sub_2 = np.array([np.cos(2*np.pi*i*__hz+0.28*2*np.pi) for i in t])

        __hz = hz
        self.sin_2 = np.array([np.sin(2*np.pi*i*__hz+0.56*2*np.pi) for i in t])
        self.cos_2 = np.array([np.cos(2*np.pi*i*__hz+0.56*2*np.pi) for i in t])

        __hz = hz + 0.2
        self.sin_add_2 = np.array([np.sin(2*np.pi*i*__hz+0.84*2*np.pi) for i in t])
        self.cos_add_2 = np.array([np.cos(2*np.pi*i*__hz+0.84*2*np.pi) for i in t])


class CrossCorrelation(object):
    '''Simple CCA between reference signal and samples'''
    def __init__(self, signal_sample, ref_signals, num, double):
        self.__double = double
        self.signal = signal_sample
        self.reference = ref_signals
        self.num_sec = num
        score_sub = max(
            np.corrcoef(self.reference.sin_sub, self.signal)[0, 1],
            np.corrcoef(self.reference.cos_sub,  self.signal)[0, 1])
        score_ext = max(
            np.corrcoef(self.reference.sin, self.signal)[0, 1],
            np.corrcoef(self.reference.cos, self.signal)[0, 1])
        score_add = max(
            np.corrcoef(self.reference.sin_add, self.signal)[0, 1],
            np.corrcoef(self.reference.cos_add, self.signal)[0, 1])
        self.score = (score_sub, score_ext, score_add)
        if self.__double:
            score_sub_2 = max(
                np.corrcoef(self.reference.sin_sub_2, self.signal)[0, 1],
                np.corrcoef(self.reference.cos_sub_2,  self.signal)[0, 1])
            score_ext_2 = max(
                np.corrcoef(self.reference.sin_2, self.signal)[0, 1],
                np.corrcoef(self.reference.cos_2, self.signal)[0, 1])
            score_add_2 = max(
                np.corrcoef(self.reference.sin_add_2, self.signal)[0, 1],
                np.corrcoef(self.reference.cos_add_2, self.signal)[0, 1])
            self.score_2 = (score_sub_2, score_ext_2, score_add_2)
