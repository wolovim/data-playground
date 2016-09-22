from flask import Flask
app = Flask(__name__)

@app.route('/titanic')
def titanic():
  return 'titanic data coming soon'

if __name__ == '__main__':
  app.run()
