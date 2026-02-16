import uproot
import numpy as np
import copy
from tqdm import tqdm
import json
import socket
import os
import functools
import operator
import importlib.util

def readSim(path, find_key=None):

    tree = uproot.open(path)

    data = []
    labels = []

    for key0 in tree.keys():
        if find_key is None:
            for key in tree[key0].items():
                try:
                    data.append(np.array(tree[key0][key[0]].array(library="np"))[0].tolist())
                    labels.append(key[0])
                except Exception as e:
                    continue
        else:
            for key in tree[key0].items():
                if(find_key in key[0]):
                    try:
                        data.append(np.array(tree[key0][key[0]].array(library="np"))[0].tolist())
                        labels.append(key[0])
                    except Exception as e:
                        continue
    
    return labels, data


def readoutSectors(digitLabels, digitData, labels = ['mTimeStamp', 'mCharge', 'mCRU', 'mRow', 'mPad']):

    digitsSectors = list()

    masks = np.zeros((len(labels),len(digitLabels)))

    for i, var in enumerate(labels):
        for j, label in enumerate(digitLabels):
            masks[i][j] = (var in label)
    
    output_data = list()

    for m in masks[np.sum(masks,axis=1)>0].astype(bool):
        output_data.append(functools.reduce(operator.iconcat, digitData[m], []))

    return np.array(labels), np.array(output_data)


def dataCreator(digitsAcc, digitsLabels, mcnative_data, mcnative_labels, number_of_samples=10000, timewindow=100, padwindow=20):
    
    mask_sector = digitsLabels=='mCRU'
    mask_row = digitsLabels=='mRow'
    mask_pad = digitsLabels=='mPad'
    mask_time = digitsLabels=='mTimeStamp'
    mask_charge = digitsLabels=='mCharge'
    
    mc_sector = mcnative_labels=='native_sector'
    mc_row = mcnative_labels=='native_row'
    mc_pad = mcnative_labels=='native_pad'
    mc_deltapad = mcnative_labels=='native_sigmapad'
    mc_time = mcnative_labels=='native_time'
    mc_deltatime = mcnative_labels=='native_sigmatime'
    mc_qtot = mcnative_labels=='native_qtot'
    mc_qmax = mcnative_labels=='native_qmax'
    mc_nclusters = mcnative_labels=='native_nclusters'
    mc_event = mcnative_labels=='native_event'
    
    full_data = np.zeros((7,36,152,120,int(np.max(digitsAcc[mask_time]))))
    input_data = np.zeros((number_of_samples, 1, padwindow, timewindow))
    output_data = np.zeros((number_of_samples, 6, padwindow, timewindow))
    
    digitsAcc[mask_sector] = np.floor(digitsAcc[mask_sector]/10.)
    
    startpad = np.random.randint(0,padwindow, size=number_of_samples)
    starttime = np.random.randint(0,timewindow, size=number_of_samples)
    
    for elem in digitsAcc.T:
        try:
            full_data[0, int(elem[mask_sector][0]), int(elem[mask_row][0]), int(elem[mask_pad][0]), int(elem[mask_time][0])] = int(elem[mask_charge][0])
        except:
            continue
        
    for elem in mcnative_data:
        try:
            full_data[1, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_qmax][0])
            full_data[2, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_qtot][0])
            full_data[3, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_deltapad][0])
            full_data[4, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_deltatime][0])
            full_data[5, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_nclusters][0])
            full_data[6, int(elem[mc_sector][0]), int(elem[mc_row][0]), int(elem[mc_pad][0]), int(elem[mc_time][0])] = int(elem[mc_event][0])
        except:
            continue
    
    shuffeled_digits = np.random.permutation(digitsAcc.T)
    
    counter = 0
    for entry in shuffeled_digits:
        if counter>=number_of_samples:
            break
        else:
            try:
                query = full_data[0,int(entry[mask_sector][0]),int(entry[mask_row][0]),(int(entry[mask_pad][0])-startpad[counter]):(int(entry[mask_pad][0])-startpad[counter]+padwindow),(int(entry[mask_time][0])-starttime[counter]):(int(entry[mask_time][0])-starttime[counter]+timewindow)]
                if (np.sum(query) != 0):
                    input_data[counter][0] = query
                    for i in range(6):
                        output_data[counter][i] = full_data[i+1,int(entry[mask_sector][0]),int(entry[mask_row][0]),(int(entry[mask_pad][0])-startpad[counter]):(int(entry[mask_pad][0])-startpad[counter]+padwindow),(int(entry[mask_time][0])-starttime[counter]):(int(entry[mask_time][0])-starttime[counter]+timewindow)]
                    counter += 1
                else:
                    continue
            except:
                continue
    
    return input_data, output_data


def dataCreatorFloat(digitsAcc, digitsLabels, mcnative_data, mcnative_labels, number_of_samples=10000, timewindow=100, padwindow=20):
    
    mask_sector = digitsLabels=='mCRU'
    mask_row = digitsLabels=='mRow'
    mask_pad = digitsLabels=='mPad'
    mask_time = digitsLabels=='mTimeStamp'
    mask_charge = digitsLabels=='mCharge'
    
    mc_sector = mcnative_labels=='native_sector'
    mc_row = mcnative_labels=='native_row'
    mc_pad = mcnative_labels=='native_pad'
    mc_time = mcnative_labels=='native_time'
    
    full_data = np.zeros((2,36,152,140,int(np.max(digitsAcc[mask_time]))+1))
    input_data = np.zeros((number_of_samples, 1, padwindow, timewindow))
    output_data = np.zeros((number_of_samples, padwindow, timewindow, 10))
    
    digitsAcc[mask_sector] = np.floor(digitsAcc[mask_sector]/10.)
    
    startpad = np.random.randint(0,padwindow, size=number_of_samples)
    starttime = np.random.randint(0,timewindow, size=number_of_samples)
    
    index_list = np.zeros((2,4), dtype=int)
    
    sorting_indices = np.arange(0,5, dtype=int)
    for i, idx_mask in enumerate([mask_sector, mask_row, mask_pad, mask_time]):
        index_list[0][i] = sorting_indices[idx_mask][0]
        
    sorting_indices = np.arange(0,10, dtype=int)
    for i, idx_mask in enumerate([mc_sector, mc_row, mc_pad, mc_time]):
        index_list[1][i] = sorting_indices[idx_mask][0]
    
    
    ### Indexing MAGIC!
    full_data[0][tuple((digitsAcc[index_list[0]].astype(int)).tolist())] = digitsAcc[mask_charge]
    full_data[1][tuple((mcnative_data.T[index_list[1]].astype(int)).tolist())] = np.arange(len(mcnative_data))
    
    shuffeled_digits = np.random.permutation(digitsAcc[index_list[0]].astype(int).T)
    
    counter = 0
    for entry in shuffeled_digits:
        if counter>=number_of_samples:
            break
        else:
            try:
                query = full_data[:,entry[0],entry[1],(entry[2]-startpad[counter]):(entry[2]-startpad[counter]+padwindow),(entry[3]-starttime[counter]):(entry[3]-starttime[counter]+timewindow)]
                if (np.sum(query[0]) != 0):
                    input_data[counter,0] = query[0]
                    output_data[counter][query[1]!=0] = mcnative_data[query[1][query[1]!=0].astype(int)]
                    counter += 1
                else:
                    continue
            except:
                continue
    
    return input_data, output_data


def dataCreatorSequence(digitsAcc, digitsLabels, mcnative_data, mcnative_labels, number_of_samples=10000, timewindow=100, padwindow=20, number_of_clusters=15):
    
    mask_sector = digitsLabels=='mCRU'
    mask_row = digitsLabels=='mRow'
    mask_pad = digitsLabels=='mPad'
    mask_time = digitsLabels=='mTimeStamp'
    mask_charge = digitsLabels=='mCharge'
    
    mc_sector = mcnative_labels=='native_sector'
    mc_row = mcnative_labels=='native_row'
    mc_pad = mcnative_labels=='native_pad'
    mc_time = mcnative_labels=='native_time'
    mc_save = mc_row + mc_pad
    #mc_save = ((mcnative_labels!='native_sector')*(mcnative_labels=='native_row')*(mcnative_labels!='native_event')*(mcnative_labels!='native_nclusters'))
    
    full_data = np.zeros((2,36,152,140,int(np.max(digitsAcc[mask_time]))+1))
    input_data = np.zeros((number_of_samples, 1, padwindow, timewindow))
    output_data = np.zeros((number_of_samples, number_of_clusters, np.sum(mc_save)))
    
    digitsAcc[mask_sector] = np.floor(digitsAcc[mask_sector]/10.)
    
    startpad = np.random.randint(0,padwindow, size=len(mcnative_data))
    starttime = np.random.randint(0,timewindow, size=len(mcnative_data))
    
    subtract_pos = np.zeros((len(mcnative_data),np.sum(mc_save)))
    subtract_pos[mcnative_labels[mc_save]=='native_pad'] = mcnative_data['native_pad']-startpad
    subtract_pos[mcnative_labels[mc_save]=='native_time'] = mcnative_data['native_time']-starttime
    
    index_list = np.zeros((2,4), dtype=int)
    
    sorting_indices = np.arange(0,5, dtype=int)
    for i, idx_mask in enumerate([mask_sector, mask_row, mask_pad, mask_time]):
        index_list[0][i] = sorting_indices[idx_mask][0]
        
    sorting_indices = np.arange(0,10, dtype=int)
    for i, idx_mask in enumerate([mc_sector, mc_row, mc_pad, mc_time]):
        index_list[1][i] = sorting_indices[idx_mask][0]
    
    
    ### Indexing MAGIC!
    full_data[0][tuple((digitsAcc[index_list[0]].astype(int)).tolist())] = digitsAcc[mask_charge]
    full_data[1][tuple((mcnative_data.T[index_list[1]].astype(int)).tolist())] = np.arange(len(mcnative_data))
    
    shuffeled_digits = np.random.permutation(digitsAcc[index_list[0]].astype(int).T)
    
    counter = 0
    for entry in shuffeled_digits:
        if counter>=number_of_samples:
            break
        else:
            try:
                query = full_data[:,entry[0],entry[1],(entry[2]-startpad[counter]):(entry[2]-startpad[counter]+padwindow),(entry[3]-starttime[counter]):(entry[3]-starttime[counter]+timewindow)]
                if (np.sum(query[0]) != 0):
                    input_data[counter,0] = query[0]
                    mask_output = query[1]!=0
                    fill_length = np.sum(mask_output)
                    output_data[counter][:fill_length] = mcnative_data[query[1][mask_output].astype(int), mc_save]
                    counter += 1
                else:
                    continue
            except:
                continue
    
    return input_data, output_data


def writeToPy(path, data, labels):

    ### Saving to files
    np.save(path+".npy", data)
    np.savetxt(path+".txt", labels, delimiter=" ", fmt="%s")


def readFromPy(pathLabels, pathData):

    return np.loadtxt(pathLabels, dtype=str), np.load(pathData)

def machinePaths():
    
    configs = {
        "lxbk0552": {
            "path": "/data.local1/csonnab",
            "dataPath": "/lustre/alice/users/csonnab"
        },

        "virgo": {
            "path": "/lustre/alice/users/csonnab",
            "dataPath": "/lustre/alice/users/csonnab/PhD/jobs/clusterization/NN/training_data"
        },

        "macOS": {
            "path": "/Users/jarvis/alice",
            "dataPath": "/Users/jarvis/cernbox/data_storage"
        },

        "linux_home": {
            "path": "/home/chris/alice",
            "dataPath": "/home/chris/cernbox/data_storage"
        },
        
        "epn": {
            "path": "/scratch/csonnabe",
            "dataPath": "/scratch/csonnabe"
        }
    }
    
    hostname = socket.gethostname()
    if "0552" in hostname:
        return configs["lxbk0552"]
    elif "Jarvis"==hostname:
        return configs["linux_home"]
    elif ("m2" in hostname.lower()) or ("mbp2" in hostname.lower()) or ("macbook" in hostname.lower()):
        return configs["macOS"]
    elif "lxbk" in hostname:
        return configs["virgo"]
    elif "login.internal" in hostname: #EPNs
        return configs["epn"]
    else:
        print("Hostname not in options, hostname = " + str(hostname))
        
def load_attr_from_file(filepath, attr_name):
    spec = importlib.util.spec_from_file_location("dynamic_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)

    # Example usage:
    # load_config_fn = load_function_from_file(load_config, "load_config")
    # cfg = load_config_fn()
