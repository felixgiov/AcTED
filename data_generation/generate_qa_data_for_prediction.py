import itertools
import os
import spacy
import argparse
import re
import random

from nltk.corpus import stopwords

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help = "Sentence-Event pairs files dir")
parser.add_argument("-o", "--output_dir", help = "Output dir")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

n = 50

sp = spacy.load('en_core_web_sm')

"""
Find and read input files
"""

sent_event_pairs_files = []
for file in os.listdir(input_dir):
    if 'sent_event_pairs' in file:
        sent_event_pairs_files.append(os.path.join(input_dir, file))

event_sents_map = {}
for file in sent_event_pairs_files:
    with open(file, 'r') as reader:
        for event in reader:
            sent = event.split('\t')[0]
            event = event.split('\t')[1].replace('\n', '')
            if event not in event_sents_map.keys():
                event_sents_map[event] = []
            event_sents_map[event].append(sent)

input_files = []
for file in os.listdir(input_dir):
    if 'all_srl_NEW' in file:
        input_files.append(input_dir+file)

events = []
for file in input_files:
    with open(file, 'r') as reader:
        for event in reader:
           events.append(event.replace('\n', ''))

# Remove duplicates and sort events
events_sorted = sorted(list(set(events)))


"""
Clean the data and remove some noises
"""

def peek(iterable):
    """
    Check if generator is empty.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)

stop_words = list(stopwords.words('english'))

events_cleaned = []
events_cleaned_count = {}
new_event_sents_map = {}

# Words to remove from the events, e.g., pronouns, etc.
words_to_remove = [' i ', ' me ', ' my ', ' myself ', ' we ', ' our ', ' ours ', ' ourselves ', ' you ', ' your ',
                   ' yours ', ' yourself ', ' yourselves ', ' he ', ' him ', ' his ', ' himself ', ' she ', ' her ',
                   ' hers ', ' herself ', ' it ', ' its ', ' itself ', ' they ', ' them ', ' their ', ' theirs ',
                   ' themselves ', ' a ', ' an ', ' the ', ' this ', ' that ', ' these ', ' those ', ' I ']

for event in events_sorted:

    event = event.replace('\n', '')
    if event in event_sents_map.keys():
        sent_count = len(event_sents_map[event])
    else:
        sent_count = 0

    # Filter out events that begin with uppercase.
    if not event[0].isupper():
        
        # Remove some words from the event.
        event = ' ' + event + ' '
        for word in words_to_remove:
            event = event.replace(word, ' ')

        event = event.strip()
        event = re.sub(' +', ' ', event)

        # Deconstruct the event.
        event_words = event.split(' ')
        beg_part = event_words[0]
        mid_part = event_words[1:-1]
        end_part = event_words[-1]

        if len(event_words) == 1:
            mid_part = ''
            end_part = ''
        elif len(event_words) == 2:
            mid_part = ''

        # Lemmatize the event's verb.
        verb = sp(beg_part)
        verb_lemma = verb[0].lemma_
        verb_all_noun_chunk = verb.noun_chunks

        res = peek(verb_all_noun_chunk)
        if res is None:
            if end_part in stop_words:
                end_part = ''

            # Reconstruct the event.
            mid = ' '.join(mid_part)
            event_new = verb_lemma.strip() + ' ' + mid.strip() + ' ' + end_part.strip()
            event_new = re.sub(' +', ' ', event_new)
            event_new = event_new.strip()
            events_cleaned.append(event_new)

            if event_new not in events_cleaned_count.keys():
                events_cleaned_count[event_new] = sent_count
            else:
                events_cleaned_count[event_new] += sent_count

            if event in event_sents_map.keys():
                for sent in event_sents_map[event]:
                    if event_new not in new_event_sents_map.keys():
                        new_event_sents_map[event_new] = []
                    new_event_sents_map[event_new].append((sent, event_new.replace('\n', '')))


"""
For each event, sample n number of sentences, then generate QA problems.
"""

event_n_sents = []  # events with at least n sentences
for event in events_cleaned_count.keys( ):
    if events_cleaned_count[event] >= n:
        event_n_sents.append(event)

event_n_sents = sorted(list(set(event_n_sents)))

event_n_sents_sampled = []
for event in event_n_sents:
    sent_sampled = random.sample(new_event_sents_map[event], n)
    event_n_sents_sampled.extend(sent_sampled)

dur_units = ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades']

qa_data = []
for sent, event in event_n_sents_sampled:
    for unit in dur_units:
        qa_data.append('{}\tHow long does it take to {}?\t{}\tno'.format(sent, event, unit))


"""
Write output to a file
"""

with open(os.path.join(output_dir, 'event_sents_pairs.tsv'), 'w') as writer:
    for line in event_n_sents_sampled:
        writer.write(line+'\n')

with open(os.path.join(output_dir, 'event_qa_to_predict.tsv'), 'w') as writer:
    for line in qa_data:
        writer.write(line+'\n')
