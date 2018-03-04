import sys
import open_bci_ganglion as bci
import tkinter as tk
import multiprocessing as mp
import time


class Imp_Check(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.button = tk.Button(self, text='Start',command=self.start).pack(side='right')
        self.button = tk.Button(self, text='Stop',command=self.stop).pack(side='right')

        self.UPDATE_RATE = 2 # in miliseconds ##FIXME: for bigger values (100+) application freezes

        self.label1 = tk.Label(self, width=80)
        self.label1.pack(side="top", fill="x")
        self.label2 = tk.Label(self, width=80)
        self.label2.pack(side="top", fill="x")
        self.label3 = tk.Label(self, width=80)
        self.label3.pack(side="top", fill="x")
        self.label4 = tk.Label(self, width=80)
        self.label4.pack(side="top", fill="x")
        self.label5 = tk.Label(self, width=80)
        self.label5.pack(side="top", fill="x")

        self.__channel1 = 0
        self.__channel2 = 0
        self.__channel3 = 0
        self.__channel4 = 0
        self.__channel5 = 0

        self.label1.configure(text="Channel 1: %s kOhm" % self.__channel1)
        self.label2.configure(text="Channel 2: %s kOhm" % self.__channel2)
        self.label3.configure(text="Channel 3: %s kOhm" % self.__channel3)
        self.label4.configure(text="Channel 4: %s kOhm" % self.__channel4)
        self.label5.configure(text="Reference: %s kOhm" % self.__channel5)

        self.streaming = mp.Event()
        self.terminate = mp.Event()

        self.prcs = mp.Process(target=self.acquire)

        self.queue = mp.Queue()
        self.queue2 = mp.Queue()
        self.queue3 = mp.Queue()
        self.queue4 = mp.Queue()

        self.prcs.start()
        root.update()
    def acquire(self,state): # single time activation
        def handle_sample(sample): # sampling function / single process
            ''' __.imp_data for 5 elements array with impedance values / __.channel_data for channel data. '''
            self.channel1 = sample.channel_data[0]
            self.channel2 = sample.channel_data[1]
            self.channel3 = sample.channel_data[2]
            self.channel4 = sample.channel_data[3]
            #self.channel5 = sample.channel_data[4]   #      turned off for channel data
            self.queue.put(self.channel1)
            self.queue2.put(self.channel2)
            self.queue3.put(self.channel3)
            self.queue4.put(self.channel4)
            #self.queue5.put(self.channel5)

            if self.board.streaming:
                self.streaming.set()

            if self.terminate.is_set():
                self.streaming.clear()
                self.board.stop()
                #self.board.disconnect()



        self.board = bci.OpenBCIBoard(impedance=True, port="d2:b4:11:81:48:ad",timeout=5)
        self.board.start_streaming(handle_sample)

        self.board.disconnect()


    def update(self):
        self.after(self.UPDATE_RATE, self.update)
        if not self.queue.empty():
            self.__channel1 = self.queue.get()
            self.__channel2 =  self.queue2.get()
            self.__channel3 =  self.queue3.get()
            self.__channel4 =  self.queue4.get()
        #   self.__channel5 =  self.queue5.get()            ## turned off for channel data

        self.label1.configure(text="Channel 1:  %0.2f  kOhm" % self.__channel1)
        self.label2.configure(text="Channel 2:  %0.2f  kOhm" % self.__channel2)
        self.label3.configure(text="Channel 3:  %0.2f  kOhm" % self.__channel3)
        self.label4.configure(text="Channel 4:  %0.2f  kOhm" % self.__channel4)
        #self.label5.configure(text="Reference: %s kOhm" % self.__channel5)

    def start(self):
        self.update()
        if not self.prcs.is_alive():
            self.prcs = mp.Process(target=self.acquire)
            self.prcs.start()
        root.update()

    def stop(self):
        self.terminate.set()
        if self.prcs.is_alive():
            self.prcs.join()
            self.prcs.terminate()
            self.streaming = mp.Event()
            self.terminate = mp.Event()
        root.update()
if __name__ == "__main__":
    root = tk.Tk()
    Imp_Check(root).pack(fill="both", expand=True)
    root.mainloop()
