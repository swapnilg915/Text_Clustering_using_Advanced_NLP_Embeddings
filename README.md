# Clustering text documents using advanced NLP embeddings 

Cluster your documents/sentences into different clusters/groups.
It uses 
1. TF-IDF
2. google's state of the art (SOTA) Word2vec model
3. google's state of the art (SOTA) BERT model

To change the embedding technique, edit it in the dataset json file under dataset directory named: clustering_20newsgroup_training_data.json

setup the project environment
1. Create virtual environment using the command : python3.7 -m venv clustering_docs_env_3.7
2. Activate the virtual environment using : source clustering_docs_env_3.7/bin/activate
3. Run the above command to install setup tools : python3 -m pip install setuptools pip install -U wheel
4. Install all the required python packages using : python3 -m pip install -r requirements.txt
5. Run the flask API : python3 clustering_api.py
6. In browser run: http://0.0.0.0:5001/apidocs
7. enter the absolute path of your dataset json file in the swagger. 

Note: The dataset used in this project is 20 newsgroup dataset which can be easily access using sklearn or tensorflow libraries. 
This dataset is stored in dataset directory in the format required to run this project. 

Start BERT service:
1. Download pre-trained bert model of your choice from : https://github.com/google-research/bert#pre-trained-models
2. start BERT server using this command: bert-serving-start -model_dir /home/swapnil/Projects/Embeddings/BERT/cased_L-12_H-768_A-12 -num_worker=4

Download google's pre-trained word2vec model from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

References:
https://github.com/hanxiao/bert-as-service
https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
https://blog.eduonix.com/artificial-intelligence/clustering-similar-sentences-together-using-machine-learning/


