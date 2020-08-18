# Image_captioning_with_deployment
## Motivation
We must first understand how important this problem is to real world scenarios. Let’s see few applications where a solution to this problem can be very useful.
1. Self driving cars — Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self driving system.
2. Aid to the blind — We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then the text to voice. Both are now famous applications of Deep Learning. Refer this link where its shown how Nvidia research is trying to create such a product.
3. CCTV cameras are everywhere today, but along with viewing the world, if we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.
Automatic Captioning can help, make Google Image Search as good as Google Search, as then every image could be first converted into a caption and then search can be performed based on the caption.

## Data Collection
There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc.
But for the purpose of this case study, I have used the Flickr 8k dataset which you can download by [filling this form](http://shannon.cs.illinois.edu/DenotationGraph/) provided by the University of Illinois at Urbana-Champaign.

## Requirements
1. Tensorflow
2. Keras
3. Numpy
4. h5py
5. Pandas
6. Nltk


## Understanding the data set
If you have downloaded the data from the link that I have provided, then, along with images, you will also get some text files related to the images. One of the files is “Flickr8k.token.txt” which contains the name of each image along with its 5 captions. We can read this file as follows:
```
101654506_8eb26cfb60.jpg#0	A brown and white dog is running through the snow .
101654506_8eb26cfb60.jpg#1	A dog is running in the snow
101654506_8eb26cfb60.jpg#2	A dog running through snow .
101654506_8eb26cfb60.jpg#3	a white and brown dog is running through a snow covered field .
101654506_8eb26cfb60.jpg#4	The white and brown dog is running over the surface of the snow .

1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
```

## Steps to execute
1. After extracting the data, execute the Image_captioning_preprocessing.ipynb file by locating the file directory and execute. This file adds "start " and " end" token to the training and testing text data also generates image encodings by feeding the image to ResNet50 model. On execution the file creates new pickled files 'corpus','tokenizer','features'.
```
descriptions['101654506_8eb26cfb60'] = ['A brown and white dog is running through the snow .', 'A dog is running in the snow', 'A dog running through snow .', 'a white and brown dog is running through a snow covered field .', 'The white and brown dog is running over the surface of the snow .']
```

3. Execute the 'Training and visualization.ipyb' file . The variable will denote the number of epochs for which the model will be trained. The models will be saved in the Output folder in this directory.

4. After training execute "Prediction.py" for generating a caption of an image. Pass the extension of the image along with the name of the image file for example, "python test.py beach.jpg". The image file must be present in the test folder.

NOTE - You can skip the training part by directly downloading the weights and model file and placing them in the Output folder since the training part wil take a lot of time if working on a non-GPU system. A GTX 1050 Ti with 4 gigs of RAM takes around 10-15 minutes for one epoch.

## Output
The output of the model is a caption to the image and a python library called pyttsx which converts the generated text to audio

## Results
Following are a few results obtained after training the model for 70 epochs.










## References
#### NIC Model
[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
#### Data
https://illinois.edu/fb/sec/1713398
#### Code reference
https://github.com/anuragmishracse/caption_generator
