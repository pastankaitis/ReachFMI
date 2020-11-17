from fmpy import *
from fmpy.util import *
import shutil
import matplotlib.pylab as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import os

path = 'fmus'
files = os.listdir(path)

def genRnds(statespace,totalComps) :
    rnds = {}

    # d2 rnds per comparison per dimension
    for v in statespace.keys():
        rnds[v] = []

        for c in range (0,totalComps*2) :
            rnds[v].append(random.uniform(statespace[v]['low'], statespace[v]['high']))


    return rnds

def listFMUs () :

    for f in files:
        print('#####################################################')

        dump(path + '/' + f)

    return 0

def listFMUvars (fmu) :
    vars = []

    md = read_model_description(path + '/' + fmu)
    for v in md.modelVariables:
        if(v.causality == 'output') :
            vars.append(v.name)
    return vars

#TODO get statespace from file
def getStateSpace(fmu) :
    vars = listFMUvars(fmu)
    ss = {}
    for v in vars :
        ss[v] = {}

        print('enter lower bound for ' + v + ':')
        ss[v]['low'] = int(input())

        print('enter upper bound for ' + v + ':')
        ss[v]['high'] = int(input())

    return ss


def normal_dist(x, mean, sd):
    prob_density = (1 / (2 * np.pi * sd ** 2)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

def uniformStateSpace(statespace,stop,numComps,numLipschitz,fmu):
    # extract the FMU to a temporary directory
    inputs = list(statespace.keys())
    fmu = path + '/' + fmu
    unzipdir = extract(fmu)
    model_description = read_model_description(unzipdir)
    fmu_instance = instantiate_fmu(unzipdir, model_description, 'CoSimulation')

    maxLips = []
    for l in range(0, numLipschitz):
        lips = []
        for s in range (0, numComps):
            start_values = {0: {}, 1: {}}

            for i in inputs:
                start_values[0][i] = random.uniform(statespace[i]['low'], statespace[i]['high'])  # 1st selected value
                start_values[1][i] = random.uniform(statespace[i]['low'], statespace[i]['high'])  # 1st selected value

            fmu_instance.reset()
            result = simulate_fmu(unzipdir,
                                  start_values=start_values[0],
                                  model_description=model_description,
                                  fmu_instance=fmu_instance,
                                  stop_time=stop)

            end_values = {0: {}, 1: {}}

            for i in inputs:
                end_values[0][i] = result[i].tolist()[-1]

            fmu_instance.reset()

            result = simulate_fmu(unzipdir,
                                  start_values=start_values[1],
                                  model_description=model_description,
                                  fmu_instance=fmu_instance,
                                  stop_time=stop)

            dist = {'s': 0, 'e': 0}

            for i in inputs:
                end_values[1][i] = result[i].tolist()[-1]

                #Each input is a new dimension therefor always perpendicular with current max dist so can use pythag to update max distance
                dist['s'] = np.power(start_values[0][i] - start_values[1][i], 2) + dist['s']
                dist['e'] = np.power(end_values[0][i] - end_values[1][i], 2) + dist['e']

            lips.append(np.sqrt(dist['e']) / np.sqrt(dist['s']))

        maxLips.append(max(lips))

    return maxLips

def uniformStateSpaceFixedRnd(rnds,stop,numComps,numLipschitz,fmu):
    # extract the FMU to a temporary directory
    inputs = list(rnds.keys())
    fmu = path + '/' + fmu
    unzipdir = extract(fmu)
    model_description = read_model_description(unzipdir)
    fmu_instance = instantiate_fmu(unzipdir, model_description, 'CoSimulation')
    rndNum = {}
    for i in inputs:
        rndNum[i] = 0

    maxLips = []
    for l in range(0, numLipschitz):
        lips = []
        for s in range (0, numComps):
            start_values = {0: {}, 1: {}}

            for i in inputs:
                start_values[0][i] = rnds[i][rndNum[i]]  # 1st selected value
                start_values[1][i] = rnds[i][rndNum[i] + 1]  # 1st selected value
                rndNum[i] = rndNum[i] + 2

            fmu_instance.reset()
            result = simulate_fmu(unzipdir,
                                  start_values=start_values[0],
                                  model_description=model_description,
                                  fmu_instance=fmu_instance,
                                  stop_time=stop)

            end_values = {0: {}, 1: {}}

            for i in inputs:
                end_values[0][i] = result[i].tolist()[-1]

            fmu_instance.reset()

            result = simulate_fmu(unzipdir,
                                  start_values=start_values[1],
                                  model_description=model_description,
                                  fmu_instance=fmu_instance,
                                  stop_time=stop)

            dist = {'s': 0, 'e': 0}

            for i in inputs:
                end_values[1][i] = result[i].tolist()[-1]

                #Each input is a new dimension therefor always perpendicular with current max dist so can use pythag to update max distance
                dist['s'] = np.power(start_values[0][i] - start_values[1][i], 2) + dist['s']
                dist['e'] = np.power(end_values[0][i] - end_values[1][i], 2) + dist['e']

            lips.append(np.sqrt(dist['e']) / np.sqrt(dist['s']))

        maxLips.append(max(lips))

    return maxLips

def lipotim ():

    for f in files :

        statespace = getStateSpace(f)

        rnds = genRnds(statespace,64*64)

        lips = uniformStateSpaceFixedRnd(rnds,3,64,64,f)

        lips.sort()

        mean = np.mean(lips)
        sd = np.std(lips)
        pdf = normal_dist(lips, mean, sd)

        # Plotting the Results
        plt.plot(lips, pdf, color='red', label='med sample med Lipschitz')

        lips = uniformStateSpaceFixedRnd(rnds,3,32,128,f)

        lips.sort()

        mean = np.mean(lips)
        sd = np.std(lips)
        pdf = normal_dist(lips, mean, sd)

        # Plotting the Results
        plt.plot(lips, pdf, color='blue', label='low sample high Lipschitz')

        lips = uniformStateSpaceFixedRnd(rnds,3,128,32,f)

        lips.sort()

        mean = np.mean(lips)
        sd = np.std(lips)
        pdf = normal_dist(lips, mean, sd)

        # Plotting the Results
        plt.plot(lips, pdf, color='green', label='high sample low Lipschitz')


        plt.xlabel('Data points')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    lipotim()