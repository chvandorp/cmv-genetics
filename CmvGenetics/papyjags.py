#!/usr/bin/env python
import subprocess
import numpy
import os.path

## aux functions
def flatten(xss): return [x for xs in xss for x in xs]
def fst((a, _)): return a
def snd((_, b)): return b


## functions to simulate the R-dump-file
def RdumpScalar(name, x):
    string = "`%s` <- %s\n"%(name, (str(x) if not numpy.isnan(x) else 'NA'))
    return string

def RdumpVector(name, xs):
    xs_str = [str(x) if not numpy.isnan(x) else "NA" for x in xs]
    string = "`%s` <- c(%s)\n"%(name, ", ".join(xs_str))
    return string
    
def RdumpMatrix(name, xss):
    n = len(xss)
    m = len(xss[0])
    xs = flatten(xss)
    xs_str = [str(x) if not numpy.isnan(x) else "NA" for x in xs]
    string = "`%s` <- structure(c(%s), .Dim=c(%d, %d))\n"%(name, ", ".join(xs_str), m, n)
    return string

## functions to create the necessary files for JAGS
def mkJagsDataFile(path, fileBaseName, pd):
    datFileName = "{0}{1}.dat".format(path, fileBaseName)
    datFileHandle = open(datFileName, 'w')
    for p, d in pd.items():
        data = numpy.array(d)
        if len(data.shape) == 0: ## data is a scalar
            datFileHandle.write(RdumpScalar(p, data))
        elif len(data.shape) == 1: ## data is a vector
            datFileHandle.write(RdumpVector(p, data))
        elif len(data.shape) == 2: ## data is a matrix
            datFileHandle.write(RdumpMatrix(p, data))
    datFileHandle.close()
    return datFileName

def mkJagsScriptFile(path, fileBaseName, pars, iter, warmup, thin, chains):
    jgsFileName = "{0}{1}.jgs".format(path, fileBaseName)
    jgsFileHandle = open(jgsFileName, 'w')
    jgsFileHandle.write("cd {}\n".format(path))
    jgsFileHandle.write("model in {}.bug\n".format(fileBaseName))
    jgsFileHandle.write("data in {}.dat\n".format(fileBaseName))
    jgsFileHandle.write("compile, nchains({})\n".format(chains))
    jgsFileHandle.write("initialize\n")
    jgsFileHandle.write("update {0}, by({1})\n".format(warmup, thin))
    for p in pars: jgsFileHandle.write("monitor {0}, thin({1})\n".format(p, thin))
    jgsFileHandle.write("update {0}, by({1})\n".format(iter, thin))
    jgsFileHandle.write("coda *, stem({}.)\n".format(fileBaseName))
    jgsFileHandle.write("exit\n")
    jgsFileHandle.close()
    return jgsFileName

def mkJagsBugFile(path, fileBaseName, model):
    bugFileName = "{0}{1}.bug".format(path, fileBaseName)
    bugFileHandle = open(bugFileName, 'w')
    bugFileHandle.write(model)
    bugFileHandle.close()
    return bugFileName

## function that runs JAGS
def runJags(path, fileBaseName):
    logfilename = "{0}{1}.log".format(path, fileBaseName)
    logfile = open(logfilename, 'w')
    errfilename = "{0}{1}.err".format(path, fileBaseName)
    errfile = open(errfilename, 'w')
    jagsCmd = "jags"
    jgsFileName = "{0}{1}.jgs".format(path, fileBaseName)
    retcode = subprocess.call([jagsCmd, jgsFileName], 
                              stdout=logfile, stderr=errfile)
    logfile.close()
    errfile.close()
    return (logfilename, errfilename)


## function that reads and parses JAGS output
def parseJagsOutput(path, fileBaseName):
    ## aux function for parsing parameter names and their index (for vectors)
    def getParNameAndIndex(pn):
        xs = pn.split('[') ## todo: use some regular expression
        pn = xs[0]
        idx = None
        if len(xs) > 1:
            idx = int(xs[1][:-1])
        return (pn, idx)
    ## aux function for collecting all elements of a parameter vector
    def sortTraces(pn, rawchain):
        indices = [getParNameAndIndex(key) for key in rawchain.keys()]
        indices = sorted([snd(x) for x in indices if fst(x) == pn])
        if len(indices) > 1:
            traces = [rawchain[pn + "[%d]"%idx] for idx in indices]
            return traces
        else:
            trace = rawchain[pn]
            return trace
        
    sams = [] ## return value (one dict for each chain)
    ## index file
    idxFileName = "{0}{1}.index.txt".format(path, fileBaseName)
    if not os.path.isfile(idxFileName): return sams
    ## else...
    idxFileHandle = open(idxFileName, 'r')
    idxs = idxFileHandle.read().split('\n')
    idxFileHandle.close()
    idxs = [row.split(' ') for row in idxs if row != '']
    idxs = dict((row[0], (int(row[1]), int(row[2]))) for row in idxs if len(row) == 3)
    ## chain files
    c = 1
    while True:
        chainFileName = "{0}{1}.chain{2}.txt".format(path, fileBaseName, c) ## TODO: multiple chains
        if not os.path.isfile(chainFileName): break
        chainFileHandle = open(chainFileName, 'r')
        rawchain = chainFileHandle.read().split('\n')
        chainFileHandle.close()
        rawchain = [row.split() for row in rawchain if row != '']
        rawchain = [row[1] for row in rawchain if len(row) == 2]
        rawchain = dict((k, [rawchain[i] for i in range(fst(idxs[k])-1, snd(idxs[k]))]) for k in idxs.keys())
        ## monitored parameter names
        monParNames = list(set([fst(getParNameAndIndex(key)) for key in rawchain.keys()]))
        sams += [dict((pn, sortTraces(pn, rawchain)) for pn in monParNames)]
        c += 1
    return sams

def calcWAIC(loglikes, verbose=False):
    """returns WAIC, WAIC_se, p_waic_hat, lpd_hat"""
    loglikes = [map(float, lls) for lls in loglikes]
    lpd_hat_vec = [numpy.log(numpy.mean(map(numpy.exp, lls))) for lls in loglikes]
    p_waic_hat_vec = [numpy.var(lls, ddof=1) for lls in loglikes]
    elpd_waic_hat_vec = [x-y for x, y in zip(lpd_hat_vec, p_waic_hat_vec)]
    WAIC_vec = [-2*x for x in elpd_waic_hat_vec]
    WAIC = numpy.sum(WAIC_vec)
    WAIC_se = numpy.sqrt(len(WAIC_vec)) * numpy.std(WAIC_vec)
    p_waic_hat = numpy.sum(p_waic_hat_vec)
    lpd_hat = numpy.sum(lpd_hat_vec)
    if verbose:
        print "WAIC = %f"%WAIC
        print "standard error = %f"%WAIC_se
        print "effective number of parameters = %f"%p_waic_hat
        print "log pointwise predictive density = %f"%lpd_hat
    return (WAIC, WAIC_se, p_waic_hat, lpd_hat)

def mergeChains(chains):
    """takes a list of dictionaries, and merges them"""
    sams = chains[0]
    pars = sams.keys()
    for chain in chains[1:]:
        for p in pars:
            sams[p] += chain[p]
    return sams
    

## a jags model object

class JagsModel:
    """An object representing a JAGS model"""
    def __init__(self, model_code="", file_name=None, model_name="anon_model", path="/tmp/"):
        if file_name is None:
            self.model_code = model_code
        else:
            fileHandle = open(path + file_name, 'r')
            self.model_code = fileHandle.read()
            fileHandle.close()
        self.model_name = model_name
        self.path = path
        ## try loading the samples from disk (empy list if does not exist yet)
        self.sams = parseJagsOutput(self.path, self.model_name)
    def __str__(self): return "JAGS model '{0}'\n{1}".format(self.model_name, self.model_code)
    def __repr__(self): return self.__str__()
    def sampling(self, 
                 data={},
                 pars=[],
                 chains=1,
                 iter=1000,
                 warmup=1000,
                 thin=1,
                 verbose=False):
        ## prepare files
        datFileName = mkJagsDataFile(self.path, self.model_name, data)
        jgsFileName = mkJagsScriptFile(self.path, self.model_name, pars, iter, warmup, thin, chains)
        bugFileName = mkJagsBugFile(self.path, self.model_name, self.model_code)
        ## run the JAGS model
        logFileName, errFileName = runJags(self.path, self.model_name)
        if verbose:
            logFileHandle = open(logFileName, 'r')
            errFileHandle = open(errFileName, 'r')
            print "log:\n------------\n{}".format(logFileHandle.read())
            print "errors:\n---------\n{}".format(errFileHandle.read())
        ## parse the model output
        self.sams = parseJagsOutput(self.path, self.model_name)
        return self.sams
    
def calcDeltaWAIC(jm1, jm2, key1="log_like", key2=None, c1=0, c2=None, verbose=False):
    ## WAIC vec for model 1
    loglikes1 = [map(float, lls) for lls in jm1.sams[c1][key1]]
    lpd_hat_vec1 = [numpy.log(numpy.mean(map(numpy.exp, lls))) for lls in loglikes1]
    p_waic_hat_vec1 = [numpy.var(lls, ddof=1) for lls in loglikes1]
    elpd_waic_hat_vec1 = [x-y for x, y in zip(lpd_hat_vec1, p_waic_hat_vec1)]
    WAIC_vec1 = [-2*x for x in elpd_waic_hat_vec1]
    ## WAIC vec for model 2
    if key2 is None: key2 = key1 ## assume that keys are equal
    if c2 is None: c2 = c1 ## idem for chain index
    loglikes2 = [map(float, lls) for lls in jm2.sams[c2][key2]]
    lpd_hat_vec2 = [numpy.log(numpy.mean(map(numpy.exp, lls))) for lls in loglikes2]
    p_waic_hat_vec2 = [numpy.var(lls, ddof=1) for lls in loglikes2]
    elpd_waic_hat_vec2 = [x-y for x, y in zip(lpd_hat_vec2, p_waic_hat_vec2)]
    WAIC_vec2 = [-2*x for x in elpd_waic_hat_vec2]
    ## compute the differences
    DeltaWAIC_vec = [x-y for x, y in zip(WAIC_vec1, WAIC_vec2)]
    DeltaWAIC = numpy.sum(DeltaWAIC_vec)
    DeltaWAIC_se = numpy.sqrt(len(DeltaWAIC_vec)) * numpy.std(DeltaWAIC_vec)
    if verbose:
        print "Delta WAIC = {0} (se = {1})".format(DeltaWAIC, DeltaWAIC_se)
    return (DeltaWAIC, DeltaWAIC_se)

    
    
## run module as script...
if __name__ == "__main__":
    print "pyjags is a simple interface between JAGS and python"