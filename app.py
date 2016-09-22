from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/titanic')
def titanic():
  return render_template('titanic.html')

if __name__ == '__main__':
  app.run()
