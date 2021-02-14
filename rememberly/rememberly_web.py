from flask import Flask, render_template, request, redirect, session, url_for
app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/', methods=["GET", "POST"])
def startpage():
	if request.method == 'POST':
		session['inputText'] = request.form['inputText']
		return redirect(url_for('analyze'))
	return render_template('interface.html')

@app.route('/text')
def analyze():
	if 'inputText' in session:
		return render_template('interface_with_text.html', inputText=session["inputText"])
	return render_template('interface_with_text.html')
