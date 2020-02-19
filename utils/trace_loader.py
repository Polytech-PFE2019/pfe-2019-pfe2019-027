import json
import os
import numpy as np

class TraceLoader:
    def __init__(self,pathname):
        self.path_to_traces=pathname
        self.trace_paths=os.listdir(pathname)

    def __len__(self):
        return len(self.trace_paths)

    def __getitem__(self,i):
        return np.array(json.load(open(self.path_to_traces+"/"+self.trace_paths[i])))
