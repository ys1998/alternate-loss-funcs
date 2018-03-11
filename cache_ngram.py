"""
Script to cache popular ngram occurences.
"""
import argparse
from six.moves import cPickle
import os

def main(args):
	dataset = args.dataset
	data_dir = args.data_dir
	cntr = 0
	bigram_count = {}
	for split in ['train', 'valid']:
		filepath = os.path.join(data_dir, dataset + '.' + split + '.txt')
		print "Reading %s ..." % filepath
		with open(filepath, 'r') as f:
			text = f.read().split(' ')

		# Bigrams
		print "Counting bigrams ..."
		for word1, word2 in zip(text[:-1], text[1:]):
			if word1 not in bigram_count or word2 not in bigram_count[word1]:
				bigram_count[word1] = {word2 : 1}
			else:
				bigram_count[word1][word2] += 1

	order = ["%d %s %s" % (cnt, word1, word2) for word1, x in bigram_count.items() for word2, cnt in x.items()]
	print "Total bigrams ... %d" % len(order)
	order = sorted(order, key=lambda x: x[0], reverse=True)
	print "Popular bigrams %s ..."
	print "\n".join(order[:50])
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the input files')
	parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
	args = parser.parse_args()
	main(args)