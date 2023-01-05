import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help = "McTACO dataset dir")
parser.add_argument("-o", "--output_dir", help = "Output dir")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

filenames = ['dev_3783', 'test_9442']

for filename in filenames:
    data = []
    with open(os.path.join(input_dir, filename + '.tsv'), 'r') as reader:
        for row in reader:
            item = row.split('\t')
            type = item[4]
            if type == 'Event Duration\n':
                data.append(row)

    with open(os.path.join(output_dir, filename + '_duration.tsv'), 'w') as writer:
        for item in data:
            writer.write(item)

    # Split dev_3783 into 80% train and 20% dev for hold-out validation
    if filename == 'dev_3783':

        data_grouped_by_question = {}
        for line in data:
            item = line.split('\t')
            context = item[0]
            question = item[1]
            key = context + '\t' + question
            if key not in data_grouped_by_question:
                data_grouped_by_question[key] = []
            data_grouped_by_question[key].append(line)

        dev_size = int(0.2 * len(data_grouped_by_question))

        data_grouped_list = list(data_grouped_by_question.items())
        random.shuffle(data_grouped_list)

        dev = data_grouped_list[:dev_size]
        train = data_grouped_list[dev_size:]

        with open(os.path.join(output_dir, filename + '_duration_train.tsv'), 'w') as writer:
            for question in train:
                for line in question[1]:
                    writer.write(line)

        with open(os.path.join(output_dir, filename + '_duration_dev.tsv'), 'w') as writer:
            for question in dev:
                for line in question[1]:
                    writer.write(line)