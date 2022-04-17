from cProfile import run
from distutils.log import debug
from flask import Flask, render_template, request
import test as lstm 

app=Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        note = request.form["keynote"]
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def results():
    body = request.form["start_string"]
    data = lstm.generate_text(body, generation_length = 2000)
    return data

    



if __name__ == "__main__":
    app.run(debug=True)