# Neural Network Dogs Breeds Classifier
> Hacked the resnet to predict 120 classes
---
usage : 
- py dogs.py pilou.jpg # => breed + probability
- py dogs.py &;#runs flask on 127.0.0.1:8080 in background, model gets preloaded => better response times

then perform your postdata using curl such as :

- curl -k 127.0.0.1:8080 -F "input=unexpected";#returns ko
- curl -k 127.0.0.1:8080 -F "filepath=pilou.jpg";#samoyed
- curl -k 127.0.0.1:8080 -F "dogPicture=@pilou.jpg";#samoyed


<img src='https://1.x24.fr/a/jupyter/dogs6/resnet50Trimmed_e_120cm1.webp' style='max-width:70vw;max-height:70vh'/>
---
based on stanford-dogs-dataset
todo : add the original work notebook and presentation
---
<center>&copy; 2020 <a href='//alpow.fr#o:mlgithub'>Alpow</a> 🗲☻</center>