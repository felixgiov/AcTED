import os
import argparse

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
