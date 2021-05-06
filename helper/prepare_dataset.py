import json
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
sentences_lst= dataset.data
labels=list([int(num)for num in dataset.target])
num_clusters = len(dataset.target_names)

required_json = {"TrainingData":sentences_lst, "labels":labels,"NumberOfClusters":num_clusters, "Language":"en", "BotId":"bot_1"}
with open("datasets/clustering_20newsgroup_training_data.json", "w+") as fs:
	fs.write(json.dumps(required_json, indent=4))
	print("\n created json successfully!")