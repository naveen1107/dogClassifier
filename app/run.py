from flask import Flask, url_for, request, render_template, flash, redirect
from werkzeug import secure_filename
import os
from keras import backend as K

IMG_FOLDER = 'static/img'

from model import my_dog_detector

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # save file in temp folder
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['IMG_FOLDER'], filename))
        return redirect(url_for('result', image=filename))
        
    return render_template('index.html')

@app.route('/delete/<image>') 
def delete(image):
    # call to delete selected 
    if request.method == 'POST':
        to_delete = os.path.join(app.config['IMG_FOLDER'], image)

@app.route('/result/<image>')
def result(image):
    print('processing image')
    
    # getting image dog breed and output string
    pred_breed, pred_string = my_dog_detector(os.path.join(app.config['IMG_FOLDER'], image))
    pred_breed = pred_breed.split(':')[-1].strip()
    print('perd_breed: ', pred_breed)
    print('pred_string: ', pred_string)
    if ':' not in pred_string:
        pred_breed = 'Neither_Dog_Nor_Human'
    #print('curr dir: ', os.getcwd())
    img_ = os.path.join('img', image)
    ref_ = os.path.join('img/Sample_breed_img', pred_breed+'.jpg')
    #print(ref_)
    #print('img ref: ', './Sample_breed_img/'+ pred_breed +'.jpg')
    print('img: ', img_)
    #print('url img: ', url_for(img_))
    #print('url img_ref', url_for(img_ref))
    return render_template('result.html', pred_string=pred_string, pred_breed=pred_breed, img_file = img_, img_ref_file=ref_)

if __name__ == '__main__':
    app.run()

    
