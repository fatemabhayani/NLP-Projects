import os
import numpy as np
import string

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    # add s and /s to the sequence
    r.insert(0, '<s>')
    h.insert(0, '<s>')
    r.append('</s>')
    h.append('</s>')
    # initialize lev matrix
    R = np.empty((len(r), len(h)), dtype=int)
    B = np.empty((len(r), len(h)), dtype=object)
    # fill 
    R[0] = np.arange(len(h))
    R[:, 0] = np.arange(len(r))
    # initialize backtrace'
    for i in range(len(h)):
        B[0, i] = [0, R[0, i], 0]
    for i in range(len(r)):
        B[i, 0] = [0, 0, R[i, 0]]
    # loop
    for i in range(1, len(r)):
        for j in range(1, len(h)):
            if r[i] == h[j]:
                R[i, j] = int(R[i - 1, j -1])
                B[i, j] = list(B[i - 1, j - 1])
            else:
                possible_outcomes = [R[i - 1, j] + 1, R[i - 1, j - 1] + 1, R[i, j - 1] + 1]
                R[i, j] = min(possible_outcomes)
                if R[i, j] == R[i - 1, j] + 1:
                    B[i, j] = list(B[i - 1, j])
                    B[i, j][2] = B[i, j][2] + 1 # deletion += 1
                elif R[i, j] ==  R[i - 1, j - 1] + 1:
                    B[i, j] = list(B[i - 1, j - 1])
                    B[i, j][0] = B[i, j][0] + 1 # substitution += 1
                elif R[i, j] == R[i, j - 1] + 1: 
                    B[i, j] = list(B[i, j - 1]) 
                    B[i, j][1] = B[i, j][1] + 1 # substitution += 1
    WER = R[len(r) - 2, len(h) - 2]/(len(r) - 2) if (len(r) - 2) != 0 else float('inf')
    nS = B[len(r) - 2, len(h) - 2][0]
    nI = B[len(r) - 2, len(h) - 2][1]
    nD = B[len(r) - 2, len(h) - 2][2]
    return (WER, nS, nI, nD)

if __name__ == "__main__":
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            with open(os.path.join(dataDir, speaker, 'transcripts.txt')) as f:
                transcript = f.readlines()
            with open(os.path.join(dataDir, speaker, 'transcripts.Google.txt')) as f:
                google = f.readlines()
            with open(os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')) as f:
                kaldi = f.readlines()
            punc_to_remove = set(string.punctuation)
            punc_to_remove.discard('[')
            punc_to_remove.discard(']')
            for i, line in enumerate(transcript):
                line = line.lower().strip()
                for punc in list(punc_to_remove):
                    line = line.replace(punc, "")
                transcript[i] = line.split()
            for i, line in enumerate(google):
                line = line.lower().strip()
                for punc in list(punc_to_remove):
                    line = line.replace(punc, "")
                google[i] = line.split()
            for i, line in enumerate(kaldi):
                line = line.lower().strip()
                for punc in list(punc_to_remove):
                    line = line.replace(punc, "")
                kaldi[i] = line.split()
            for i in range(len(transcript)):
                print(i)
                WER, nS, nI, nD = Levenshtein(transcript[i], google[i])
                print(f"{speaker} Google {i} {WER} S:{nS} I:{nI} D:{nD}")
                WER, nS, nI, nD = Levenshtein(transcript[i], kaldi[i])
                print(f"{speaker} Kaldi {i} {WER} S:{nS} I:{nI} D:{nD}")
