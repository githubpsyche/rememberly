from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def startpage():
        return render_template('interface.html')

@app.route('/send_data', methods = ['POST'])
def get_HTML_data():
        text = request.form['textInput']
        print("Text Inputted is" + textInput)
        return json.loads(text)[0]
