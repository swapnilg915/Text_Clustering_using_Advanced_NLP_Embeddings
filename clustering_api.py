import os
import re
import json
import time
import traceback
import numpy as np

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, render_template

from training.text_clustering import ClusterText
text_clustering_obj = ClusterText()

# for reproducible results
np.random.seed(777)

app = Flask(__name__)
app.debug = True
app.url_map.strict_slashes = False
Swagger(app)


@app.route("/clustering", methods=['POST', 'OPTIONS'])
@swag_from("swagger_files/clustering_api.yml")
def clustering():
	res_json = {"result": {}, "status":"", "model_id":"", "lang":""}
	try:
		st = time.time()
		body_data = json.loads(request.get_data())
		print("\n keys received --- ",body_data.keys())

		""" read input request """ 
		training_data_path = body_data["TrainingDataPath"]
		body_data = json.load(open(training_data_path, "r"))
		text_data = body_data["TrainingData"]
		number_of_clusters = int(body_data["NumberOfClusters"])
		lang = body_data["Language"]
		model_id = body_data["ModelId"]
		labels=body_data["labels"]
		embedding_model = body_data["EmbeddingModel"]

		print("\n number of clusters --- ",number_of_clusters)
		print("\n language --- ",lang)
		print("\n model_id --- ", model_id)
		print("\n embedding_model --- ",embedding_model)

		""" train clustering model """
		clusters_dict = text_clustering_obj.main(text_data, labels, lang, embedding_model, number_of_clusters, model_id)
		res_json["result"] = json.dumps(clusters_dict, indent=4)
		res_json["model_id"] = model_id
		res_json["lang"] = lang
		res_json["status"] = "200"
		print("\n total time --- ",time.time() - st)

	except Exception as e:
		logger.error("\n Error in clustering app main() :",traceback.format_exc())
		res_json["status"] = "400"

	# return render_template("index.html")
	return jsonify(res_json)

@app.route("/clustering_results", methods=['GET', 'OPTIONS'])
def clustering_results():
	return render_template("index.html")



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5001)

	""" INPUT DATA FORMAT

	data_json = {
		"BotId":"bot_3",
		"Language":"nb",
		"number_of_clusters":292,
		"TrainingData":[
			
		]
	}

	"""