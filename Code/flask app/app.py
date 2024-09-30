from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, viswa! ' + 'viswanath D' + '7376232AL511' + 'dept of ai&ml'

if __name__ == '__main__':
    app.run(debug = True)
