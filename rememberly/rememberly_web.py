from flask import Flask, render_template, request, redirect
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def startpage():
        if request.method == 'POST':
                print(request.form['inputText'])
        return render_template('interface.html')
