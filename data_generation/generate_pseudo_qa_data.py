import os
import random
from random import randrange
import argparse

from findpeaks import findpeaks

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--event_sents_pairs_file", help = "Event-sentences pairs file path")
parser.add_argument("-p", "--predicted_qa_file", help = "Predicted QA file path")
parser.add_argument("-o", "--output_dir", help = "Output dir")
args = parser.parse_args()

event_sents_pairs_file = args.event_sents_pairs_file
predicted_qa_file = args.predicted_qa_file
output_dir = args.output_dir


"""
Read input files and generate the duration distribution for each event.
"""

event_sents_pairs = []
with open(event_sents_pairs_file, 'r') as reader:
    for line in reader:
        line_splits = line.split('\t')
        sent = line_splits[0]
        event = line_splits[1].replace('\n', '')
        event_sents_pairs.append((sent, event))

predicted_qa = []
with open(predicted_qa_file, 'r') as reader:
    for line in reader:
        predicted_qa.append(line.replace('\n', ''))

dur_dist = {}
dur_dist_by_context = {}

for i, (sent, event) in enumerate(event_sents_pairs):
    if event not in dur_dist.keys():
        dur_dist[event] = [0] * 8

    event_sent_pair = event + '\t' + sent
    if event_sent_pair not in dur_dist_by_context.keys():
        dur_dist_by_context[event_sent_pair] = [0] * 8

    for j in range(8):
        if predicted_qa[i*(j+8)] == 'yes':
            dur_dist[event][j] += 1
            dur_dist_by_context[event_sent_pair][j] = 1


"""
Find the episodic and habitual durations from the peaks and their neighbouring units from the duration distributions.
"""

neighbour_thr = 0.75

dur_units_to_index = {'seconds': 0, 'minutes': 1, 'hours': 2, 'days': 3, 'weeks': 4, 'months': 5, 'years': 6, 'decades': 7}
dur_units = dur_units_to_index.keys()

typical_duration_all = []
typical_duration_episodic = []

for i, event in enumerate(dur_dist.keys()):
    durations = dur_dist[event]

    # k predictions are removed in order to TODO
    k = 3
    durations = [unit_count - k if unit_count >= k else 0 for unit_count in durations]
    durations = [0] + durations + [0]

    multiPeak = False

    fp = findpeaks(method='topology', lookahead=1)
    results = fp.fit(durations)

    episodic_dur, episodic_dur_value, habitual_dur, habitual_dur_value = -1, -1, -1, -1

    for w, peak in enumerate(results['df']['peak']):
        if w < 4:
            if peak and results['df']['y'][w] > episodic_dur_value:
                episodic_dur = w
                episodic_dur_value = results['df']['y'][w]
        else:
            if peak and results['df']['y'][w] > habitual_dur_value:
                habitual_dur = w
                habitual_dur_value = results['df']['y'][w]

    peaks = []
    for z, peak in enumerate(results['df']['peak']):
        if peak and not results['df']['valley'][z]:
            peaks.append(z)

    peaks = [x-1 for x in peaks] 

    peak_count = len(peaks)
    if peak_count > 0:
        last_peak_idx = peaks[-1]

        if peak_count > 1:
            episodic_peak_idx = peaks[0]
            habitual_peak_idx = peaks[-1]

            episodic_units_list = [0] * 8
            episodic_units_list[episodic_peak_idx] = 1

            episodic_peak_unit = dur_units[episodic_peak_idx]
            habitual_peak_unit = dur_units[habitual_peak_idx]

            # Find the neighbouring units after checking if the peak is not on the left-most or the right-most of the distribution
            if episodic_peak_idx != 0 and durations[episodic_peak_idx - 1] >= neighbour_thr * durations[episodic_peak_idx]:
                    episodic_peak_left_neighbour = dur_units[episodic_peak_idx - 1]
                    episodic_units_list[episodic_peak_idx - 1] = 1

            if episodic_peak_idx != 7 and durations[episodic_peak_idx + 1] >= neighbour_thr * durations[episodic_peak_idx]:
                    episodic_peak_right_neigbour = dur_units[episodic_peak_idx + 1]
                    episodic_units_list[episodic_peak_idx +1 ] = 1

            if habitual_peak_idx != 0 and durations[habitual_peak_idx - 1] >= neighbour_thr * durations[habitual_peak_idx]:
                    habitual_peak_left_neighbour = dur_units[habitual_peak_idx - 1]

            if habitual_peak_idx != 7 and durations[habitual_peak_idx + 1] >= neighbour_thr * durations[habitual_peak_idx]:
                    habitual_peak_right_neigbour = dur_units[habitual_peak_idx + 1]

            typical_duration_all.append((event, durations, episodic_peak_left_neighbour, episodic_peak_unit, episodic_peak_right_neigbour,
                         habitual_peak_left_neighbour, habitual_peak_unit, habitual_peak_right_neigbour))
            typical_duration_episodic.append((event, durations, episodic_units_list))

        else:
            episodic_units_list = [0] * 8
            episodic_units_list[last_peak_idx] = 1

            peak_unit = dur_units[last_peak_idx]

            left_neighbour = ''
            right_neigbour = ''

            if last_peak_idx != 0:
                if durations[last_peak_idx - 1] >= neighbour_thr * durations[last_peak_idx]:
                    left_neighbour = dur_units[last_peak_idx - 1]
                    episodic_units_list[last_peak_idx - 1] = 1

            if last_peak_idx != 7:
                if durations[last_peak_idx + 1] >= neighbour_thr * durations[last_peak_idx]:
                    right_neigbour = dur_units[last_peak_idx + 1]
                    episodic_units_list[last_peak_idx + 1] = 1

            typical_duration_all.append((event, durations, left_neighbour, peak_unit, right_neigbour))
            typical_duration_episodic.append((event, durations, episodic_units_list))


"""
Generate the answers for the pseudo QA data
"""

dur_upper_bounds = [(60, 'seconds'), (60, 'minutes'), (24, 'hours'), (7, 'days'), (52, 'weeks'), (12, 'months'), (10, 'years'), (10, 'decades'), (10, 'centuries')]

dur_variations = [['a few seconds', 'several seconds'],
                  ['a few minutes', 'several minutes'],
                  ['a few hours', 'several hours', 'for hours'],
                  ['a few days', 'several days', 'for days'],
                  ['a few weeks', 'several weeks', 'for weeks'],
                  ['a few months', 'several months',  'for months'],
                  ['a few years', 'several years', 'for years'],
                  ['a few decades', 'several decades', 'for decades']]


def generate_answers(n, range_min, range_max, dur_unit, type):
    """
    Return a list of answers with size n for pseudo QA data.
    TODO: simplify and refactor this function, reduce loops.
    """
    answers = []
    number_selected= []

    if type == 'pos':  # Generate positive answers.
        number_1_selected = False 
        while len(answers) < n:
            
            # Randomly select which kind of answers will be generated, i.e, random numbers or random phrases from dur_variations.
            if randrange(4) in range(3):
                if int(range_max) - int(range_min) > 1:
                    number = 1
                    # Randomly select even number or multiples of 5 and check if such number hasn't been selected before.
                    while (number % 2 != 0 and number % 5 != 0) and number not in number_selected:
                        number = randrange(int(range_min), int(range_max))
                    number_selected.append(number)
                else:
                    number = randrange(int(range_min), int(range_max))

                if number == 1:
                    if not number_1_selected:
                        answers.append(str(number) + ' ' + dur_upper_bounds[dur_unit][1][:-1])
                        number_1_selected = True
                    else:
                        phrase_selected = random.choice(dur_variations[dur_unit])
                        while phrase_selected not in answers:
                            answers.append(phrase_selected)
                else:
                    answers.append(str(number)+' '+dur_upper_bounds[dur_unit][1])
            else:
                phrase_selected = random.choice(dur_variations[dur_unit])
                while phrase_selected not in answers:
                    answers.append(phrase_selected)

    else:   # Generate negative answers.
        for i in range(n):
            if randrange(5) in range(2):
                answers.append(random.choice(dur_variations[dur_unit]))
            else:
                number = 1
                while number % 2 != 0 and number not in number_selected:
                    number = randrange(1, int(dur_upper_bounds[dur_unit][0]))
                number_selected.append(number)
                answers.append(str(number) + ' ' + dur_upper_bounds[dur_unit][1])

    return answers

# Number of positive answers and negative answers to be generated.
pos_num = 3
neg_num = 4

pseudo_qa = {}

for item in typical_duration_episodic:
    event = item[0]
    episodic_peak_units = item[2]
    question = 'How long does it take to ' + event + '?'
    contexts = []

    for event in dur_dist_by_context:
        if event.startswith(event+'\t') and episodic_peak_units == dur_dist_by_context[event]:
            contexts.append(event.split('\t')[1])

    if len(contexts) >= 1:
        sampled_contexts = random.sample(contexts, 1)

        for j, context in enumerate(sampled_contexts):
            pos_features = []
            neg_features = []
            pos_index = [i for i, x in enumerate(episodic_peak_units) if x == 1]
            pos_index_w_neighbours = []

            for index in pos_index:
                pos_index_w_neighbours.append(index-1)
                pos_index_w_neighbours.append(index)
                pos_index_w_neighbours.append(index+1)
            pos_index_w_neighbours = set(pos_index_w_neighbours)
            neg_index = []

            for i in range(8):
                if i not in pos_index_w_neighbours:
                    neg_index.append(i)

            for unit_idx in pos_index:
                pos_features.extend(generate_answers(pos_num, 1, dur_upper_bounds[unit_idx][0], unit_idx, 'pos'))

            for unit_idx in neg_index:
                neg_features.extend(generate_answers(neg_num, 1, dur_upper_bounds[unit_idx][0], unit_idx, 'neg'))

            pos_features = list(set(pos_features))
            neg_features = list(set(neg_features))

            if len(pos_features) > pos_num:
                pos_features = random.sample(pos_features, pos_num)

            if len(neg_features) > neg_num:
                neg_features = random.sample(neg_features, 4)

            for ans in pos_features:
                feat = context + '\t' + question + '\t' + ans + '\tyes\tEvent Duration'

                if event not in pseudo_qa:
                    pseudo_qa[event] = []
                pseudo_qa[event].append(feat)

            for ans in neg_features:
                feat = context + '\t' + question + '\t' + ans + '\tno\tEvent Duration'

                if event not in pseudo_qa:
                    pseudo_qa[event] = []
                pseudo_qa[event].append(feat)

pseudo_qa_events = list(pseudo_qa.keys())
random.shuffle(pseudo_qa_events)


"""
Split the pseudo QA data with 200 events increments and output them into files
"""

n = 200  
pseudo_qa_events_splits = []
for i in range(0, len(pseudo_qa_events), n):
    pseudo_qa_events_splits.append(pseudo_qa_events[i:i+n])

pseudo_qa_data = []
for i, split in enumerate(pseudo_qa_events_splits):
    for event in split:
        pseudo_qa_data.extend(pseudo_qa[event])

    with open(os.path.join(args.output_dir, 'pseudo_qa_{}.tsv'.format(i+1)*n), 'w') as writer:
        for line in pseudo_qa_data:
            writer.write(line)

