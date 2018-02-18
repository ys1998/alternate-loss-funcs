COUNTS = {u'M\u0101ori': 19, u'No\xebl': 1, u'\xc9cole': 1, u'D\xfcsseldorf': 1, u'Jos\xe9': 1, u'\u2011': 67, u'\u2013': 357, u'\u2212': 2, u'\u2014': 161, u'\u2019': 37, u'\u2018': 3, u'f\xfcr': 3, u'\u201d': 6, u'\u201c': 4, u'\xa3': 37, u'Sh\u014dnen': 1, u'\u20a4': 1, u'\u03b3': 1, u'Hern\xe1n': 1, u'\u03b1': 5, u'\xb0': 40, u'\u2033': 10, u'\u2032': 14, u'\xdaltimo': 1, u'Michoac\xe1n': 1, u'\xbd': 3, u'Agust\xedn': 1, u'clich\xe9d': 1, u'fianc\xe9e': 1, u'\u2044': 2, u'Tom\xe1\u0161': 3, u'G\xfcnther': 1, u'\u03b2': 9, u'n\xe9e': 2, u'Mar\xeda': 1, u'\u03bcm': 2, u'fa\xe7ade': 1, u'\xb2': 1, u'Cort\xe9s': 1, u'\u266d': 6, u'\u266f': 1, u"'\xe9tat": 1, u'\xcele': 2, u'C\xe9sar': 1}

def main():
	MAPPING = {}
	for i, t in enumerate(COUNTS.items()):
		MAPPING[t[0]] = "<unk%d>" % i

	with open('wiki/wiki.train.txt', 'r') as f:
			text = f.read().split(' ')

	for i in range(len(text)):
		if text[i] in MAPPING:
			text[i] = MAPPING[text[i]]
			print("Replaced")

	with open('wiki/wiki.train.txt', 'w') as f:
		f.write(' '.join(text))

	with open('wiki/wiki.valid.txt', 'r') as f:
			text = f.read().split(' ')

	for i in range(len(text)):
		if text[i] in MAPPING:
			text[i] = MAPPING[text[i]]
			print("Replaced")


	with open('wiki/wiki.valid.txt', 'w') as f:
		f.write(' '.join(text))

	with open('wiki/wiki.test.txt', 'r') as f:
			text = f.read().split(' ')

	for i in range(len(text)):
		if text[i] in MAPPING:
			text[i] = MAPPING[text[i]]
			print("Replaced")

	with open('wiki/wiki.test.txt', 'w') as f:
		f.write(' '.join(text))

if __name__ == '__main__':
	main()
