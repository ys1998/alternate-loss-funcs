import argparse
from six.moves import cPickle
import os

def isASCII(s):
	return all(ord(c) < 128 for c in s)

def main(args):
	MAPPING = {}
	dataset = args.dataset
	data_dir = args.data_dir
	cntr = 0
	for split in ['train', 'valid', 'test']:
		filepath = os.path.join(data_dir, dataset + '.' + split + '.txt')
		print "Mapping %s ..." % filepath
		with open(filepath, 'r') as f:
			text = f.read().split(' ')
		for idx, word in enumerate(text):
			if not isASCII(word):
				if word not in MAPPING:
					MAPPING[word] = "<unk%d>" % cntr
					cntr += 1
				text[idx] = MAPPING[word]
		print "Writing mapped data to %s ..." % filepath
		with open(filepath, 'w') as f:
			f.write(' '.join(text))

	saved_mapping = os.path.join(data_dir, 'mapping.pkl')
	print "Saving mapping to %s ..." % saved_mapping
	with open(saved_mapping, 'wb') as f:
		cPickle.dump(MAPPING, f)
		
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the input files')
	parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
	args = parser.parse_args()
	main(args)