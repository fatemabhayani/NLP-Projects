import numpy as np
import argparse
import json
import re
import string

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

bnpath = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
warpath = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
alt_path = '/u/cs401/A1/feats/Alt_IDs.txt'
right_path = '/u/cs401/A1/feats/Right_IDs.txt'
center_path = '/u/cs401/A1/feats/Center_IDs.txt'
left_path = '/u/cs401/A1/feats/Left_IDs.txt'
alt_array = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
left_array = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
right_array = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
center_array = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
#
#alt_array = np.load('C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Alt_feats.dat.npy')
#left_array = np.load('C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Left_feats.dat.npy')
#right_array = np.load('C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Right_feats.dat.npy')
#center_array = np.load('C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Center_feats.dat.npy')
#bnpath = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/Wordlists/BristolNorms+GilhoolyLogie.csv'
#warpath = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/Wordlists/Ratings_Warriner_et_al.csv'
#alt_path = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Alt_IDs.txt'
#right_path = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Right_IDs.txt'
#left_path = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Left_IDs.txt'
#center_path = 'C:/Users/Fatema/Documents/ThirdYear/CSC401/A1/feats/Center_IDs.txt'

file = open(bnpath, "r")
BN_data = file.read().split('\n')[1:-3]
file.close()
BNGL = {}
for line in BN_data:
    word = line.split(',')
    BNGL[word[1]] = (word[3], word[4], word[5])

file = open(warpath, "r")
WR_data = file.read().split('\n')[1:-1]
war = {}
for line in WR_data:
    word = line.split(',')
    war[word[1]] = (word[2], word[5], word[8])
    

alt_ids = {}
right_ids =  {}
left_ids = {}
center_ids =  {}
alt_file = open(alt_path, 'r')
left_file = open(left_path, 'r')
right_file = open(right_path, 'r')
center_file = open(center_path, 'r')
alt_data = alt_file.read().split()
left_data = left_file.read().split()
right_data = right_file.read().split()
center_data = center_file.read().split()
alt_file.close()
left_file.close()
right_file.close()
center_file.close()
for i in range(len(alt_data)):
    alt_ids[alt_data[i]] = i
for i in range(len(left_data)):
    left_ids[left_data[i]] = i
for i in range(len(right_data)):
    right_ids[right_data[i]] = i
for i in range(len(center_data)):
    center_ids[center_data[i]] = i
    
def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    feats = np.zeros(173)
    body = re.compile("(\S+)/(?=\S+)").findall(comment) # comment text
    lemma = re.compile("(?<=\S)/(\S+)").findall(comment) # POS tags
    # TODO: Extract features that rely on capitalization.
    for token in body:
        if token.isupper() and len(token) >= 3:
            feats[0] += 1
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # already lowercase
    # TODO: Extract features that do not rely on capitalization.
    body = [x.lower() for x in body]
    # Num 1st person pronouns
    step1 = 0
    step2 = 0
    step3 = 0
    for token in body:
        if token in FIRST_PERSON_PRONOUNS:
            step1 += 1
        if token in SECOND_PERSON_PRONOUNS:
            step2 += 1
        if token in THIRD_PERSON_PRONOUNS:
            step3 += 1
    feats[1] = step1
    # Num 2nd person
    feats[2] = step2
    # num 3rd person
    feats[3] = step3
    # count CC tags
    feats[4] = lemma.count('CC')
    #past tense 
    feats[5] = lemma.count('VBD')
    #future tense
    step6 = ['\'ll', 'will', 'shall', 'gonna']
    future1 = re.compile(r'\b(' + r'|'.join(step6) + r')\b').findall(comment)
    future2 = re.compile(r"go/VBG to/TO [\w]+/VB").findall(comment)
    feats[6] += len(future1) + len(future2)
    #commas
    step7 = re.compile("(?=/)[\S]+").sub('', comment)
    feats[7] = step7.count(',')
    # multicharacter
    feats[8] = len(re.findall(' \W{2,}/', comment))
    #common nouns
    feats[9] = lemma.count('NN') + lemma.count('NNS')
    #proper nouns
    feats[10] = lemma.count('NNP') + lemma.count('NNPS')
    # adverbs
    feats[11] += lemma.count('RB') + lemma.count('RBR') + lemma.count('RBS')
    # wh- words
    feats[12] += lemma.count('WP') + lemma.count('WDT') + lemma.count('WRB') + lemma.count('WP$') 
    # slang
    for word in body:
        if word in SLANG:
            feats[13] += 1
    # avg sent.
    step15 = comment.count('\n')
    feats[14] = len(body)/step15 if step15 != 0 else 0
    # avg. token length
    num_tokens = 0
    length = 0
    for token in body:
        if not set(token).issubset(set(string.punctuation)):
            num_tokens += 1
            length += len(token)
    if body != "":
        feats[15] = length/num_tokens if num_tokens != 0 else 0
    # num sentences
    feats[16] = comment.count('\n')
    # 18-29
    # for Bristol, Gilholly, and Logie norms
    AoA_score = [] 
    IMG_score = []
    FAM_score = []
    # for Warringer Norms
    V_score = []
    A_score = []
    D_score = []

    # check if token exists in the Bristol dict. If yes, add their score to list
    for token in body:
        if token in BNGL:
            AoA_score.append(BNGL[token][0])
            IMG_score.append(BNGL[token][1])
            FAM_score.append(BNGL[token][2])
        if token in war:
            V_score.append(war[token][0])
            A_score.append(war[token][1])
            D_score.append(war[token][2])
    AoA_score = np.asarray(AoA_score, np.float32)
    IMG_score = np.asarray(IMG_score, np.float32)
    FAM_score = np.asarray(FAM_score, np.float32)
    V_score = np.asarray(V_score, np.float32)
    D_score = np.asarray(D_score, np.float32)
    A_score =  np.asarray(A_score, np.float32)
    feats[17] = np.mean(AoA_score)
    feats[18] = np.mean(IMG_score)
    feats[19] = np.mean(FAM_score)
    feats[20] = np.std(AoA_score)
    feats[21] = np.std(IMG_score)
    feats[22] = np.std(FAM_score)
    feats[23] = np.mean(V_score)
    feats[24] = np.mean(A_score)
    feats[25] = np.mean(D_score)
    feats[26] = np.std(V_score)
    feats[27] = np.std(A_score)
    feats[28] = np.std(D_score)
    return feats


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    


    if comment_class == "Alt":
        comment_position = alt_ids[comment_id]
        feats = np.append(feats[:29], alt_array[comment_position])

    elif comment_class == "Center":
        comment_position = center_ids[comment_id]
        feats = np.append(feats[:29], center_array[comment_position])
 
    elif comment_class == "Left":
        comment_position = left_ids[comment_id]
        feats = np.append(feats[:29], left_array[comment_position])
       
    elif comment_class == "Right":
        comment_position = right_ids[comment_id]
        feats = np.append(feats[:29], right_array[comment_position])

    return feats
 

def main(args):

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    dict_ahh = {"Alt": 3, "Center": 1, "Left": 0, "Right": 2}
    for i, comment in enumerate(data):
        curr = extract1(comment['body'])
        comment_class = comment['cat']
        comment_id = comment['id']
        curr = extract2(curr, comment_class, comment_id)
        feats[i,:-1] = curr
        feats[i, -1] = dict_ahh[comment_class] 
    feats = feats.astype(np.float32)
    feats = np.nan_to_num(feats)
    # TODO: Use extract1 to find the first 29 features for each 
    # data point. Add these to feats.
        
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    # TODO: Iterate through comments in JSON file
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

