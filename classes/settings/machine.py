import socket

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
