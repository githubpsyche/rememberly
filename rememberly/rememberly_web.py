from flask import Flask, render_template, request, redirect
import sys
app = Flask(__name__)

@app.route('/')
def startpage():
        return render_template('interface.html')

@app.route('/data')
def get_HTML_data():
        text = request.form['textInput']
        sys.stderr.write(text)
        return redirect('/')
