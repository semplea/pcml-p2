import numpy as np
import pickle
import sys

def filterVocab(file, vocab, size, full=False):
	with open(file) as f:
		with open(vocab, "rb") as voc:
			v = pickle.load(voc)
			print(len(v))

			embeddings = np.zeros((len(v), size))
			print(embeddings.shape)

			found = {}
			for line in f:
				s = line.split(" ")
				if s[0] in v:
					found[s[0]] = True
					embeddings[v[s[0]],:] = s[1:]

			#filtered = {}
			#for w in v:
			#	if w in found:
			#		filtered[w] = v[w]

			print(len(found))
			print(embeddings.shape)

	np.save("embeddings_fasttext" + ("_full" if full else ""), embeddings)
	#pickle.dump(v, open("vocab_glove.pkl", "wb"))

if __name__ == '__main__':
	full = len(sys.argv) >= 2 and sys.argv[2] == "full"
	filterVocab("twitter-datasets/model.vec", "twitter-datasets/" + ("vocab_full" if full else "vocab") + ".pkl", 100, full)
