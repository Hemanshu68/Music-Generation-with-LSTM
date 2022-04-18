from cProfile import run
from distutils.log import debug
from flask import Flask, render_template, request
from mitdeeplearning import lab1 as mdl;

import music21 as ms
import test as lstm 

app=Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def results():
    body = request.form["start_string"]
    data = lstm.generate_text(body, generation_length = 2000)
    text  = mdl.extract_song_snippet(data)
    
    return render_template('results.html', data = text)


if __name__ == "__main__":
    app.run(debug=True)