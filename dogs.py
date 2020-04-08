#py dogs.py pilou.jpg
#mainframe inclusion
if 'MainFrame':
    import os
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    #sys.stdout = open(os.devnull, "w")
    #sys.stderr = open(os.devnull, "w")
    #}{
    requiredModules='Flask keras tensorflow webptools pysftp numpy requests wget scikit-multilearn'.split(' ')  
    if True:
      fn='modules-versions.txt'
      os.system('pip freeze > '+fn)
      #ftpput(fn)
      installed=''
      with open(fn) as f:
        installed += f.read()
          
      for module in requiredModules:
        if(module+'==' not in  installed):
          #print('Trying to install :',module)
          os.system('pip install '+module)

      os.system('pip freeze > '+fn);

    #os.system('rm -f alpow.py gv.py');os.system('wget https://alpow.fr/gv.py')os.system('wget https://alpow.fr/alpow.py');
    import gv;import alpow;from alpow import *
    SG('noClassRewrite',False);
    SG('webRepo','https://1.x24.fr/a/jupyter/');SG('sftp',{'cd':'dogs6','web':GG('webRepo'),'h':'-','u':'-','p':'-'});SG('useFTP',False);#ReadOnly
        
    import tensorflow

    def load(fn='allVars',onlyIfNotSet=1):
      fns=fn.split(',')
      for fn in fns:
        fn=fn.strip(', \n')
        ok=1
        if(len(fn)==0):
          continue
        if(onlyIfNotSet):
          if fn in globals().keys():
      #override empty lists, dict, dataframe and items      
            if type(globals()[fn])==type:
              continue;
            elif type(globals()[fn])==pd.DataFrame:
              if globals()[fn].shape[0]>0:            
                continue
            elif(type(globals()[fn])==dict):
              if(len(globals()[fn])>0):
                continue
            elif(type(globals()[fn])==list):
              if(len(globals()[fn])>0):
                continue
            elif(type(globals()[fn])==scipy.sparse.csr.csr_matrix):
              if(globals()[fn].shape[0]>0):
                continue
            elif(type(globals()[fn])==np.ndarray):
              if(globals()[fn].shape[0]>0):
                continue
      #si déjà définie, passer au prochain     
            elif(globals()[fn]):
              continue
        globals().update(alpow.resume(fn))
      #endfor fn
      return;

    #load('X_train')

    def extract(x):
      liste=list(x.keys())
      for i in liste:
        globals()[i]=x[i]
      p('extracted : ',','.join(liste))

    #jeuDonnees=compact('y_test,')
    def compact(variables):
      x={}
      for i in variables.split(','):
        x[i]=globals()[i]    
      p('compacted : ',variables)
      return x

    def loadIfNotSet(x):
      if x not in globals().keys():
        load(x)

    #les grosses variables
    ###############}{

    import numpy as np 
    import warnings,psutil

    from tensorflow import keras
    '''
    from keras.preprocessing.image import array_to_img 
    import keras.applications.resnet50
    import keras.applications.inception_v3
    import keras.applications.xception
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import VGG16
    # keras image preparation
    from keras.applications.vgg16 import preprocess_input
    # decode prediction
    from keras.applications.vgg16 import decode_predictions
    import sklearn.metrics
    import keras.applications.vgg16

    import keras.engine.sequential
    from keras.utils import to_categorical
    from keras import optimizers

    from keras.preprocessing.image import ImageDataGenerator

    #from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
    #from keras.models import Sequential
    from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
    from tensorflow.keras.models import Sequential
    '''
    import joblib
    import tensorflow

    def load_prepare_img(path_img,width=224,height=224,simple=False):  
      img_raw = load_img(path_img, target_size=(width, height)) 
      img = img_to_array(img_raw)
      if(simple):
        return preprocess_input(img)#3 dimensions for my models

      img = img[np.newaxis, :]#4 dimensions
      return preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    ####}{
    #incremental data
    #AttributeError: module 'tensorflow' has no attribute 'placeholder'
    #load('modelInfo')
    ftpls()

if 'common':
    def img2class(img,short=False):
        if(not os.path.exists(img)):
            p('sorry '+img+' does not exists')
        testPics2=np.array([load_prepare_img(i,simple=True) for i in [img]])
        predictions=mdl.predict(testPics2)
        max=np.amax(predictions)
        predictionsC=np.argmax(predictions, axis=-1) 
        decoded=globals()['raceLabel'+nbBreeds].inverse_transform(predictionsC)
        if(short):
            return decoded[0]
        return '#Result: ' + decoded[0] + ' ;with probability of  '+str(round(max,2))
    #autant le faire tourner avec Flask, non ?
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import preprocess_input

    nbBreeds='120'
    mdlname='resnet50Trimmed_e_120'
    load(mdlname+',raceLabel120')
    mdl=globals()[mdlname]

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    #cmdline not flask    
    if len(sys.argv)>1:
        print(img2class(sys.argv[1]))
        
#flask mode
if len(sys.argv)<2:
    from flask import Flask, render_template, url_for, request
    app = Flask(__name__)
    @app.route('/')
    def home():
        return """<html><head><title>Dog Breed Predictor from image</title>
<style>html{font-size:10px;background:#000 url('//x24.fr/0/b1.jpg##20200103SkiRandoClusaz') top center fixed;}/*background-size:110%;background-repeat:repeat;*/
body{height:100vh;color:#FFF;margin:0;}
body,pre{font:2rem 'Dancing Script',Assistant,roboto,calibri,corbel,verdana;}
*{transition:all .5s}
a{color:#0F0;}
a:hover{color:#FC0;}

fieldset{margin:0 2vw;background:rgba(255,255,255,0.1);}
legend{padding:0 3rem;}/*margin:auto;*/

input,textarea{width:100%;padding-left:1rem;font-size:2rem}
textarea{height:30vh;}

input[type=submit]{
    cursor:pointer;height:5vh;font-size:4rem;
}
input[type=submit]:hover{filter:invert(100%);}
h1,h2{color:#F00;margin:0;}

</style>
</head><body><center><fieldset><legend>Dog Breed Predictor from image</legend><form method='post' enctype='multipart/form-data' action='/'>Local filepath : <input name=filepath value='pilou.jpg'><br>Or upload file here<input type=file name=dogPicture><br><input type=submit value=submit></form></fieldset></center></body></html>
""";
        
    @app.route('/', methods=['POST'])
    def post():
        if (request.method == 'POST'):
        #print(('filepath' in request.form.keys()))
            if 'filepath' in request.form.keys():
                if(len(request.form['filepath'])>1):
                    return  img2class(request.form['filepath'],True)
                
            if 'dogPicture' in request.files.keys():
                file = request.files['dogPicture']
                x='uploads/'+file.filename
                file.save(x)
                return img2class(x,True)
        return 'ko'
    app.run(host='0.0.0.0', port=8081, debug=True)
    p('end')

#py dogs.py &        
#curl -k 127.0.0.1:8081 -F "input=unexpected"
#curl -k 127.0.0.1:8081 -F "filepath=pilou.jpg"
#curl -k 127.0.0.1:8081 -F "dogPicture=@pilou.jpg"










