import os
import json
import time
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

""" import gensim modules"""
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

""" import sklearn modules """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

from bert_serving.client import BertClient

from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en
stopwords_en = list(stopwords_en)
mapping_dict = {"en":[stopwords_en, spacy_en]} 

""" import scripts"""
from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()

import configuration_file as config


class ClusterText():

	def __init__(self):
		self.oov_list=[]



	def visualize_data(self, pred_dict, lang, bot_id, sentences):
		dir_name = os.path.join("static/results", str(bot_id), lang)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
			print("\n dir created --- ",dir_name)

		for k,v in pred_dict.items():
			tokens_list = " ".join([word for sent in sentences for word in sent.split()])
			wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = mapping_dict[lang][0], min_font_size = 10).generate(tokens_list)
			plt.figure(figsize = (8, 8), facecolor = None) 
			plt.imshow(wordcloud)
			plt.axis("off") 
			plt.tight_layout(pad = 0)
			fig_name= "cluster_" + str(k) + ".png"
			plt.savefig(os.path.join(dir_name, fig_name))
			# plt.clf()
		return pred_dict


	def train_model(self, sentences, text_vector, labels, number_of_clusters, lang, bot_id):
		try:
			""" training """
			st = time.time()
			model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=10, n_init=10, random_state=42)
			# km = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean')
			# km = DBSCAN(eps=0.4, min_samples=2)
			model.fit(text_vector)
			print("\n training time --- ", time.time() - st)
			print("\n labels --- ", model.labels_, len(model.labels_))

			print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, model.labels_))
			print("Completeness: %0.3f" % metrics.completeness_score(labels, model.labels_))
			""" harmonic mean of homogenity and completedness """
			print("V-measure: %0.3f" % metrics.v_measure_score(labels, model.labels_))
			print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, model.labels_))
			print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(text_vector, model.labels_, metric='euclidean'))

			""" Insights """
			order_centroids = []
			terms = []
			order_centroids = model.cluster_centers_.argsort()[:, ::-1]
			# terms = tfidf_vec.get_feature_names()
			# print("\n total number of terms --- ", len(terms))

			""" store clustering results in json file """
			pred = model.labels_
			pred_dict = defaultdict()
			for idx in range(len(pred)):
				if str(pred[idx]) not in pred_dict: pred_dict[str(pred[idx])] = []
				if sentences[idx] not in pred_dict[str(pred[idx])]: pred_dict[str(pred[idx])].append(sentences[idx])


			""" data visualization """
			pred_dict = self.visualize_data(pred_dict, lang, bot_id, sentences)

			""" write clustering results to json """
			with open("nelfo_kmeans_tfidf_res.json", "w+") as fs:
				fs.write(json.dumps(pred_dict, indent=4))
		except Exception as e:
			print("\n Error in train_model : ",e)
			print("\n Error details : ", traceback.format_exc())
		return model


	def load_word2vec(self):
		st = time.time()
		# self.model = Word2Vec.load('20_newsgroup_word2vec.model')
		self.model = KeyedVectors.load_word2vec_format(config.word2vec_model_path, limit=50000, binary=True)
		print("\n time to load the fasttext model --- ", time.time() - st)
		print("\n hello vector --- ", self.model['hello'])


	def sent_vectorizer(self, sent):
		sent_vec = np.zeros((1,300))
		numw = 0
		try:
			for w in sent.split():
				try:
					if numw == 0:
						sent_vec = self.model[w]
					else:
						sent_vec = np.add(sent_vec, self.model[w])
					numw+=1
				except:
					self.oov_list.append(w)
					pass
			sentence_vector = np.asarray(sent_vec) / numw
		except Exception as e:
			print(sent_vec, numw)
			print("\n Error in sent_vectorizer : ",e)
			print("\n Error details : ", traceback.format_exc())
		return sentence_vector


	def create_bert_vectors(self, sentences):
		print("\n inside create_bert_vectors ::: ")
		self.bert_client = BertClient()
		return self.bert_client.encode(sentences)


	def create_w2v_vectors(self, sentences):
		self.load_word2vec()
		st = time.time()
		X = [self.sent_vectorizer(sent) for sent in sentences ]
		X = np.array(X)
		print("\n shape --- ", X.shape)
		print("\n time for w2v vectorization : ", time.time() - st)
		del self.model
		return X


	def create_tfidf_vectors(self, sentences):
		try:
			tfidf_vec = TfidfVectorizer(use_idf=True, sublinear_tf=True, max_df=0.8, max_features=1000, ngram_range=(1,1), min_df=5)
			X = tfidf_vec.fit_transform(sentences).toarray()
			# total_terms = tfidf_vec.get_feature_names()
			# print("\n total number of terms in data --- ", len(total_terms))
			# X = StandardScaler().fit_transform(X)
			print("\n vector shape : ", X.shape)
		except Exception as e:
			print("\n Error in create_tfidf_vectors : ",e)
			print("\n Error details : ", traceback.format_exc())
		return X


	# def create_bert_vectors(self, sentences):
	# 	with BertClient(port=5555, port_out=5556) as bc:
	# 		doc_vecs = bc.encode(questions)

	def create_lsa_vectors(self, sentences):
		""" LSI """
		n_components = 5000
		svd = TruncatedSVD(self.n_components)
		normalizer = Normalizer(copy=False)
		self.lsa = make_pipeline(svd, normalizer)
		X = self.lsa.fit_transform(X)
		return X


	def main(self, text_data, labels, lang, embedding_model, number_of_clusters, bot_id):
		try:
			print("\n inside main ::: ")
			cleaned_sentences = cleaning_pipeline_obj.cleaning_pipeline(text_data, lang)
			if embedding_model == "tfidf": text_vector = self.create_tfidf_vectors(cleaned_sentences)
			elif embedding_model == "word2vec": text_vector = self.create_w2v_vectors(cleaned_sentences)
			elif embedding_model == "bert": text_vector = self.create_bert_vectors(cleaned_sentences)
			model = self.train_model(cleaned_sentences, text_vector, labels, number_of_clusters, lang, bot_id)
		except Exception as e:
			print("\n Error in main : ",e)
			print("\n Error details : ", traceback.format_exc())


if __name__ == "__main__":
	obj = ClusterText()
	# obj.main()

	"""
reference : https://blog.eduonix.com/artificial-intelligence/clustering-similar-sentences-together-using-machine-learning/
	"""