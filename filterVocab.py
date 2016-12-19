import numpy as np
import pickle

def filterVocab(file, vocab, size):
	with open(file) as f:
		with open(vocab, "rb") as voc:
			v = pickle.load(voc)
			print(len(v))

			glove = np.zeros((len(v), size))
			print(glove.shape)

			found = {}
			for line in f:
				s = line.split(" ")
				if s[0] in v:
					found[s[0]] = True
					glove[v[s[0]],:] = s[1:]

			#filtered = {}
			#for w in v:
			#	if w in found:
			#		filtered[w] = v[w]

			print(len(found))
			print(glove.shape)

	np.save("embeddings_glove", glove)
	#pickle.dump(v, open("vocab_glove.pkl", "wb"))

if __name__ == '__main__':
	filterVocab("twitter-datasets/glove.twitter.27B.25d.txt", "twitter-datasets/vocab.pkl", 25)