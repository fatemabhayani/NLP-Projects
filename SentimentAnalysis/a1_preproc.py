import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
        modComm = str(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(r"\s+", " ", modComm).strip()
    
    # TODO: get Spacy document for modComm
    modComm = nlp(modComm)
    # TODO: use Spacy document for modComm to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
    modComm1 = ""
    for sent in modComm.sents:
        for token in sent:
            lemma = ''
            if token.lemma_[0] == '-':
                lemma = token.text
            else:
                lemma = token.lemma_
            modComm1 += lemma + '/' + token.tag_ + " "
        modComm1 = modComm1[:-1] + '\n'
    modComm = modComm1
    return modComm


def main(args):
    allOutput = []
    print('here 3')
    print(indir)
    for subdir, dirs, files in os.walk(indir):
        print('here 4')
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            print('here 5')

            # TODO: select appropriate args.max lines
            starting = args.ID[0] % len(data)
            max_iteration = 0
            # TODO: read those lines with something like `j = json.loads(line)`
            while max_iteration < args.max:
                j = json.loads(data[starting])
            # TODO: choose to retain fields from those lines that are relevant to you
                k = {key:j[key] for key in ['id', 'body']}
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                k['cat'] = file
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
                k['body'] = preproc1(k['body'])
            # TODO: append the result to 'allOutput'
                allOutput.append(k)
                max_iteration += 1
                starting += 1
                if starting == args.max:
                    starting = 0
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
