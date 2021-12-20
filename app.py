import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import urllib
import json
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from static.cbir import ConvAutoEncoder

# create flask instance
app = Flask(__name__)

# main route
@app.route('/')
def index():
    return render_template('index.html')

# search route
@app.route('/search', methods=['POST'])
def search(): 
    if request.method == "POST":

        RESULTS_ARRAY = []
        feature_path1 = "static/modelfeatures/model2_training_3_subclass_feature.json"
        feature_path2 = "static/modelfeatures/model3_training_3_subclass_feature.json"
        feature_path3 = "static/modelfeatures/model4_training_3_subclass_feature.json"
        model_path1 = "static/model/training_model2.h5"
        model_path2 = "static/model/training_model3.h5"
        model_path3 = "static/model/training_model4.h5"
        IMAGE_SIZE = (128, 128)
        image_url = request.form.get('img')
        select_model = request.form.get('select')
        print(select_model)

        base_dataset = "static/dataset"
        class_dir = ['Normal', 'Tube', 'Effusion']
        type_dataset = ['val', 'train']
        dataset_train = []
        dataset_val = []
        for type_set in type_dataset:
            for class_item in class_dir:
                cur_dir = os.path.join(base_dataset, type_set, class_item)
                for file in os.listdir(cur_dir):
                    if type_set == 'train':
                        dataset_train.append(os.path.join(cur_dir, file))
                    else:
                        dataset_val.append(os.path.join(cur_dir, file))

        try:
            req = urllib.request.urlopen(image_url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)
            image = cv2.resize(image, IMAGE_SIZE)
            image = np.array(image).astype("float32") / 255.0
            image = tf.expand_dims(image, axis = 0)
            with open(feature_path1) as f:
                training_indexed = json.load(f)
            #auto_encoder = ConvAutoEncoder.build(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
            auto_encoder = load_model(model_path1)
            encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoded").output)
            features_retrieved = encoder.predict(image)[0]
            
            def euclidean(a, b):
                # compute and return the euclidean distance between two vectors
                return np.linalg.norm(a - b)

            def perform_search(query_features, indexed_train, max_results=5):
                retrieved = []
                for idx in range(0, len(indexed_train["features"])):
                    distance = euclidean(query_features, indexed_train["features"][idx])
                    retrieved.append((distance, idx))
                retrieved = sorted(retrieved)[:max_results]
                return retrieved

            results = perform_search(features_retrieved, training_indexed, max_results=5)
            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                result_path = dataset_train[resultID]
                RESULTS_ARRAY.append(
                    {"image": str(result_path), "score": str(score)})
			# return success
            return jsonify(results=(RESULTS_ARRAY[::-1][:5]))
        except:
            return (jsonify({image_url}), 500)

# run!
if __name__ == '__main__':
    app.run(debug=True)