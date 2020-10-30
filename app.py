import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2

from Searcher import Searcher
from ColorDescriptor import ColorDescriptor

app = Flask(__name__)
INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')
cd = ColorDescriptor()


@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = cv2.imdecode(np.fromstring(request.files['img'].read(), np.uint8), cv2.IMREAD_COLOR)
        features = cd.describe(query)
        searcher = Searcher(INDEX)
        results = searcher.search(features, 10)
        res = []
        for (score, resultID) in results:
            res.append({"Image": str(resultID), "Score": str(score)})
        context = {"images": res}
        print(context)
        return render_template('index.html', context=context)
    res = []
    context = {"images": res}
    return render_template('index.html', context=context)


if __name__ == '__main__':
    app.run('127.0.0.1', debug=True)
