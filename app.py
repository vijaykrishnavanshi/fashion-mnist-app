from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
from load import *
from skimage import color

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = os.path.basename('data')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
H = 28
W = 28

model, graph = load_graph_weights()
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index_page():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	file = request.files['fileupload']
	filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(filename)
	img = color.rgb2gray(imread(filename, mode='L'))
	img = imresize(img, (28, 28))
	img = img.reshape(1, 28, 28, 1)
	with graph.as_default():
		out = model.predict(img)
		print(out)
		print(np.argmax(out,axis=1))
		response = np.array_str(np.argmax(out,axis=1))
		return response		
		#return render_template('predicted.html')
if __name__ == "__main__":
	app.run()
