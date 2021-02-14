from flask import Flask, render_template, request, redirect, session, url_for

from Treehacks_recommendation import *
from visualisation import *
app = Flask(__name__)


app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

main_input = ""

@app.route('/', methods=["GET", "POST"])
def startpage():
	if request.method == 'POST':
		session['inputText'] = request.form['inputText']
		main_input = session['inputText']
		input_list = main_input.split(".")
		# print(input_list)
		output1 = textsummary(input_list)
		session['outputText'] = output1[0]
		searchresult = googleSearchURL(["Mozart deaf","shakespeare warwickshire"])
		session['SearchResult'] = searchresult
		return redirect(url_for('analyze'))
	return render_template('interface.html')

@app.route('/text')
def analyze():
	if 'inputText' in session:
		list1 = [[.4, .6, .7, .567, .643, .754], [.3, .23, .5, .234, .543, .435], [.4, .7, .2, .9, .65, .3456],
				 [.4, .6, .7, .567, .643, .754], [.3, .23, .5, .234, .543, .435], [.4, .7, .2, .9, .65, .3456]]
		text_units = ['book', 'read', 'wall', 'science', 'semantic', 'analysis']
		graph_url = landscape_surfaceplot(list1,text_units)

		return render_template('interface_with_text.html',
							   inputText=session["inputText"],
							   outputText=session["outputText"],
							   graph = graph_url,
							   search_result = session['SearchResult']
							   )
	return render_template('interface_with_text.html')



if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
	print(main_input)
