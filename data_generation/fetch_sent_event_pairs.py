import json
import argparse
import os
import logging
from subprocess import PIPE, Popen
from tqdm import tqdm

from nltk import Tree
from nltk.tokenize import sent_tokenize

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conceptnet_file", help = "ConceptNet file path")
parser.add_argument("-w", "--wiki_dir", help = "Wikipedia dump files dir")
parser.add_argument("-o", "--output_dir", help = "Output dir")
args = parser.parse_args()

conceptnet_path = args.conceptnet_path
wiki_path = args.wiki_path
out_path = os.path.join(args.output_dir, 'sent_event_pairs.tsv')


"""
Fetch English concepts from ConceptNet 
"""

concepts = []

with open(conceptnet_path, 'r') as reader:
    for line in reader:
        splitted = line.split('\t')
        if splitted[2].split('/')[2] == 'en':
            y = json.loads(splitted[4])
            if 'surfaceStart' in y.keys():
                concepts.append(y['surfaceStart'])

        if splitted[3].split('/')[2] == 'en':
            y = json.loads(splitted[4])
            if 'surfaceEnd' in y.keys():
                concepts.append(y['surfaceEnd'])

concepts_set = list(set(concepts))


"""
Keep only verb phrases as events 
"""

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

events = []

for i, concept in enumerate(concepts_set):
    try:
        parse = predictor.predict(sentence=concept)
        t = Tree.fromstring(parse['trees'])
        if t[0].label() == 'VP':
            events.append(concept)
    except Exception as e:
        logger.warning(e)


"""
Collect Wikipedia sentences that contain the exact events
"""

event_keywords = []
for line in events:
     # Filter out long events
    if len(line.split()) <= 3: 
        event_keywords.append(line.replace('\n', ''))


sent_event_pairs = []
for i, event in enumerate(event_keywords):

    # Fetch the first 200 sentences that contain the event.
    p1 = Popen(['grep', '-rw', wiki_path, '-e', event, '-h'], stdout=PIPE)
    p2 = Popen(['head', '-n200'], stdin=p1.stdout, stdout=PIPE)
    p1.stdout.close() 
    out, err = output = p2.communicate()

    paragraphs = output[0].decode("utf-8").split('\n')
    
    for paragraph in paragraphs:
        sents = sent_tokenize(paragraph)
        for sent in sents:
            if ' ' + event + ' ' in ' ' + sent + ' ':
                sent_event_pairs.append(sent + '\t' + event)


"""
Write output to a file
"""

with open(out_path, 'w') as writer:
    for sent_event_pair in sent_event_pairs:
        writer.write(sent_event_pair + '\n')