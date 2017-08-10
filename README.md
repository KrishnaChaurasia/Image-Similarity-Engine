# Image-Similarity-Engine
The image similarity engine finds the most matched image(s) in the data set for an input image. The engine involves the following steps:
1.	Install the Graph Lab Create Application: <br/>
  a.	To perform deep learning to build the engine <br/>
2.	Downloading the training data as the GraphLab SFrame <br/>
  a.	http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_train_data <br/>
3.	Extract deep features of images using the Transfer Learning techniques <br/>
  a.	Import the deep learning model from the following link https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45 <br/>
  b.	Extract the deep features of the images <br/>
4.	Use the KNN model to find similar images <br/>
