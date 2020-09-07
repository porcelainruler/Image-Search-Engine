import os
from flask import Flask, render_template, request, jsonify
from skimage import io
import cv2

from searchEngine import searchEngine
from ImageDescriptor import ImageDescriptor
import config

# create flask instance
app = Flask(__name__)


# Getting Index Location
INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')


# main route
@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':

        results = list()

        img_url = request.form.get('img')

        try:
            # Get Image Feature Extractor
            img_desc = ImageDescriptor(config.bins)

            query = io.imread(img_url)
            query = (query * 255).astype("uint8")
            (r, g, b) = cv2.split(query)
            query = cv2.merge([b, g, r])
            features = img_desc.featureExtracter(query)

            # perform the search
            # INDEX = config.idxPath  || Most Prob
            searcher = searchEngine(INDEX)
            results = searcher.search(features)

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                results.append({"Image": str(resultID), "Score": str(score)})
            # return success
            return jsonify(results=(results[:3]))
        except:
            # return error
            jsonify({"sorry": "Sorry, no results! Please try again."}), 500

    return render_template('index.html')


# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
