from flask import Flask, render_template, url_for, request
import summarizer as smry
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    data = request.form.get('text', 'Error Reading Text Data' )
    option = request.form.get('options', 'option1')
    if(option == 'option1'):
        summary = smry.lr(data)
    elif(option == 'option2'):
        summary = smry.tr(data)
    elif(option == 'option3'):
        summary = smry.ti(data)
    elif(option == 'option4'):
        summary = smry.deep(data)
    return render_template('result.html',summarized = summary)

if __name__ == "__main__":
    app.run(debug=True)