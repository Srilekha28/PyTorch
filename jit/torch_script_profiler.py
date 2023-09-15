import numpy as np
import torch
from time import perf_counter
import matplotlib.pyplot as plt

class TorchScriptProfiler:
    traced_model = None
    native_model = None
    args = []
    def __init__(self, native_model, traced_model, *args):
        self.native_model = native_model
        self.traced_model = traced_model
        for arg in args:
            self.args.append(arg)

    def run(self, executions=1, generatePlot=False):
        meanTimeArr = []
        for numTimesExecution in range(1, executions + 1):
            print("Running PyTorch with number of executions: ", numTimesExecution)
            pyTorchMeanTime = np.mean(
                [timer(native_model, *args) for _ in range(numTimesExecution)])
            print("Running TorchScript with number of executions: ", numTimesExecution)
            torchScriptMeanTime = np.mean(
                [timer(traced_model, *args) for _ in range(numTimesExecution)])
            meanTimeArr.append([numTimesExecution, pyTorchMeanTime, torchScriptMeanTime])

        if generatePlot:
            self.generatePlot(meanTimeArr)
        return meanTimeArr

    def generatePlot(self, arr):
        # Define data values in array
        arr = np.array(meanTimeArr)

        print(np.shape(arr), type(arr), arr, sep='\n')

        # Plot a simple line chart
        plt.plot(arr[:, 0], arr[:, 1], 'g', label='PyTorch Mean Runtime')

        # Plot another line on the same chart/graph
        plt.plot(arr[:, 0], arr[:, 2], 'r', label='TorchScript Mean Runtime')

        plt.legend()
        plt.show()

    def timer(f, *args):
        start = perf_counter()
        f(*args)
        return (1000 * (perf_counter() - start))
