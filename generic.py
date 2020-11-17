from fmpy import *
from fmpy.util import *
import shutil
import matplotlib.pylab as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

path = 'projects'
projects = os.listdir(path)

def genRnds(statespace,totalComps) :
    rnds = {}

    # d2 rnds per comparison per dimension
    for v in statespace.keys():
        rnds[v] = []

        for c in range (0,totalComps*2) :
            rnds[v].append(random.uniform(statespace[v]['low'], statespace[v]['high']))


    return rnds


def listProjects(pr) :
    proj = []
    for dir in projects :
        try :
            project = os.listdir(path + '/' + dir)

            if pr:
                print(
                    '##################################################################################################')
                if project.__contains__('README.md'):
                    dump(path + '/' + dir + '/README.md')
                else:
                    print('Project Name: ' + dir)
                    listFMUs(dir)
                print(
                    '##################################################################################################')

            proj.append(dir)

        except :
            print(dir + ' is not a directory')

    return proj


def listFMUs (project,pr) :
    fmus = []
    try:
        readfmus = os.listdir(path + '/' + project + '/FMUs')
        for f in readfmus:
            if f.endswith(".fmu"):
                if pr:
                    print(
                        '-------------------------------------------------------------------------------------------------')
                    dump(path + '/' + project + '/FMUs' + '/' + f)
                    listFMUvars(project, f, pr)
                fmus.append(f)
            else:
                print(f + " is not an fmu")
    except :
        print(project + ' has no FMUs folder')

    return fmus

def listFMUvars (project,fmu,pr) :
    vars = []

    md = read_model_description(path + '/' + project + '/FMUs/' + fmu)
    for v in md.modelVariables:
        if v.causality == 'output':
            vars.append(v.name)
        if pr:
            print(v.name + ": " + v.causality)
    return vars

def getStateSpace(project, fmu) :
    vars = listFMUvars(project, fmu, False)
    ss = {}
    with open(path + '/' + project + '/initialss.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader :
            if row.__len__() == 3:
                if(vars.__contains__(str(row[0]))) :
                    ss[str(row[0])] = {'low': int(float(row[1])+0.5), 'high': int(float(row[2])+0.5)}
            else:
                if str(row[0]) == 'endTime':
                    ss['stopTime'] = float(row[1])
                if str(row[0]) == 'stepSize':
                    ss['stepSize'] = float(row[1])

    if not ss.keys().__contains__('stopTime'):
        md = read_model_description(path + '/' + project + '/FMUs/' + fmu)
        ss['stopTime'] = md.defaultExperiment.stopTime

    if not ss.keys().__contains__('stepSize'):
        md = read_model_description(path + '/' + project + '/FMUs/' + fmu)
        ss['stepSize'] = md.defaultExperiment.stepSize


    varsSet = set(vars)

    for v in varsSet.difference(ss.keys()):
        ss[v] = {}

        print('enter lower bound for ' + v + ':')
        ss[v]['low'] = int(input())

        print('enter upper bound for ' + v + ':')
        ss[v]['high'] = int(input())

    return ss

def uniformStateSpaceFMU(numComps,numLipschitz,project,fmu):
    statespace = getStateSpace(project,fmu)
    stop = statespace['stopTime']
    statespace.pop('stopTime')
    stepSize = statespace['stepSize']
    print("Step size: " + str(stepSize) + ". End time: " + str(stop))
    statespace.pop('stepSize')

    inputs = list(statespace.keys())

    # extract the FMU to a temporary directory
    fmu = path + '/' + project + '/FMUs/' + fmu
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
                                  fmu_instance=fmu_instance,
                                  stop_time=stop,
                                  step_size=stepSize)

            end_values = {0: {}, 1: {}}

            for i in inputs:
                end_values[0][i] = result[i].tolist()

            fmu_instance.reset()

            result = simulate_fmu(unzipdir,
                                  start_values=start_values[1],
                                  fmu_instance=fmu_instance,
                                  stop_time=stop,
                                  step_size=stepSize)

            dist = {}

            for i in inputs:
                end_values[1][i] = result[i].tolist()


            for ts in range (0,end_values[0][i].__len__()):
                dist[ts] = 0
                temp = []
                for i in inputs:
                    #Each input is a new dimension therefor always perpendicular with current max dist so can use pythag to update max distance
                    dist[ts] = np.power(end_values[0][i][ts] - end_values[1][i][ts], 2) + dist[ts]

                    temp.append(np.sqrt(dist[ts]) / np.sqrt(dist[0]))
                lips.append(temp)

        temp = []

        for ts in range(0,lips.__len__()):
            temp.append(max(lips[ts]))

        maxLips.append(temp)
    try:
        fmus = os.listdir(path + '/' + project + '/Results')
    except:
        print('No results folder. Creating new results folder')
        os.mkdir(path + '/' + project + '/Results')

    time = datetime.now()
    extension = "th"
    if time.strftime("%d") == "1" or time.strftime("%d") == "21" or time.strftime("%d") == "31" :
        extension = "st"
    else :
        if time.strftime("%d") == "2" or time.strftime("%d") == "22":
            extension = "nd"

    dir = path + '/' + project + '/Results/'

    while True:
        try :
            os.mkdir(dir + time.strftime("%Y/%b/%d") + extension + time.strftime("/%H-%M"))
            break
        except:
            try:
                os.mkdir(dir + time.strftime("%Y/%b/%d") + extension)
            except:
                try:
                    os.mkdir(dir + time.strftime("%Y/%b"))
                except:
                    os.mkdir(dir + time.strftime("%Y"))

    dir = dir + time.strftime("%Y/%b/%d") + extension + time.strftime("/%H-%M")

    with open (dir + '/max_lips.csv','w', newline='') as csvfile:
        file = csv.writer(csvfile)
        for ts in range(0,maxLips.__len__()):
            row = [ts]
            for l in maxLips[ts]:
                row.append(l)
            file.writerow(row)

    with open (dir + '/reachss.csv','w', newline='') as csvfile:
        file = csv.writer(csvfile)

        start_values = {}

        for i in statespace.keys():
            start_values[i] = (statespace[i]['low'] + statespace[i]['high']) / 2

        fmu_instance.reset()
        result = simulate_fmu(unzipdir,
                              start_values=start_values,
                              fmu_instance=fmu_instance,
                              stop_time=stop,
                              step_size=stepSize)

        end_values = {}

        for i in inputs:
            end_values[i] = result[i].tolist()

        lip = []
        for ts in range(0,maxLips[0].__len__()):
            temp = []
            for numMax in range(0,maxLips.__len__()):
                temp.append(maxLips[numMax][ts])
            lip.append(max(temp))

        maxDistToEdge = 0
        for i in inputs:
            maxDistToEdge = max([0.5 * (statespace[i]['high'] - statespace[i]['low']), maxDistToEdge])

        for i in inputs:
            lower = []
            upper = []
            timeStamps = []
            for ts in range(0,end_values[i].__len__()):
                timeStamps.append(stepSize * ts)
                multi = lip[ts] * maxDistToEdge
                lower.append(end_values[i][ts] - multi)
                upper.append(end_values[i][ts] + multi)
                file.writerow((i, ts, str(end_values[i] - multi), str(end_values[i] + multi)))

            plt.plot(timeStamps, lower, 'r')  # plot velocity1
            plt.plot(timeStamps, upper, 'g')
            plt.plot(timeStamps, end_values[i], 'b')
            plt.xlabel('time')
            plt.ylabel(i)
            plt.show()

    return 0

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

if __name__ == '__main__':
    proj = listProjects(False)
    for dir in proj:

        fmus = listFMUs(dir, True)
        for fmu in fmus:
            uniformStateSpaceFMU(64, 64, dir, fmu)
