from flask import Flask, render_template, request, redirect, session, url_for
import numpy as np

from Treehacks_recommendation import *
from visualisation import *
from LandscapeModel.landscape import Landscape
from Preprocessing import *

# setup flask
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
main_input = ""

# set up the model
lsr = Landscape("stsb-distilbert-base")
sentence = None
text_units = None


@app.route('/', methods=["GET", "POST"])
def startpage():
    if request.method == 'POST':
        session['inputText'] = request.form['inputText']

        main_input = session['inputText']
        input_list = main_input.split(".")
        # print(input_list)
        output1 = textsummary(input_list)
        session['outputText'] = output1[0]
        searchresult = googleSearchURL(["Mozart deaf", "shakespeare warwickshire"])
        session['SearchResult'] = searchresult
        return redirect(url_for('analyze'))
    return render_template('interface.html')


@app.route('/text')
def analyze():
    if 'inputText' in session:
        text = session['inputText']

        # preprocess the input
        sentence, text_units = cycles_and_units(text)

        # generate probabilities
        lsr(text_units)
        while 1:
            try:
                lsr.cycle()
            except IndexError:
                break
        output_probs = lsr.output_probabilities(1.0)[1].T

        # get single list of text_units
        recall_cycles = []
        for reading_cycle in text_units:
            recall_cycles += [*reading_cycle]
        graph_url = landscape_surfaceplot(output_probs, recall_cycles)

        return render_template('interface_with_text.html',
                               inputText=session["inputText"],
                               outputText=session["outputText"],
                               graph=graph_url,
                               search_result=session['SearchResult']
                               )
    return render_template('interface_with_text.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
    print(main_input)
