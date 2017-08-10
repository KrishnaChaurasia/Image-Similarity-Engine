# coding: utf-8
import graphlab
graphlab.canvas.show()

graphlab.canvas.set_target('ipynb')

# Download the training data
gl_img = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_train_data')
gl_img

# Display the first five images
gl_img['image'][0:5].show()

# Resize to view it large
graphlab.image_analysis.resize(gl_img['image'][2:3], 96,96).show()

# Load our input image
img = graphlab.Image('C:\Users\Dell\Desktop\pexels-photo.jpg')
ppsf = graphlab.SArray([img])
ppsf = graphlab.image_analysis.resize(ppsf, 32,32)
ppsf.show()

# Create the SFrame of the image to extract its features
ppsf = graphlab.SFrame(ppsf).rename({'X1': 'image'})

# Loading the deep learning model from the GraphLab website
deep_learning_model = graphlab.load_model('https://static.turi.com/products/graphlab-create/resources/models/python2.7/imagenet_model_iter45')
# Extract the deep features of our image
ppsf['deep_features'] = deep_learning_model.extract_features(ppsf)

# Give a label and id to the input image
ppsf['label'] = 'me'
gl_img['id'].max()
ppsf['id'] = 50000

# Join everything together
labels = ['id', 'image', 'label', 'deep_features']
part_train = gl_img[labels]
new_train = part_train.append(ppsf[labels])
new_train.tail()

ppsf.show()

# Use the KNN model to find similar images
knn_model = graphlab.nearest_neighbors.create(new_train,features=['deep_features'], label='id')

# A test image
cat_test = new_train[-2:-1]
graphlab.image_analysis.resize(cat_test['image'], 96,96).show()

# Find its nearest matches
sim_frame = knn_model.query(cat_test)

# Top five nearest matches
def reveal_my_twin(x):
    return gl_img.filter_by(x['reference_label'],'id')
spirit_animal = reveal_my_twin(knn_model.query(cat_test))
spirit_animal['image'].show()

# Now use the input image as the test image
me_test = new_train[-1:]
graphlab.image_analysis.resize(me_test['image'], 96,96).show()

sim_frame = knn_model.query(me_test)
sim_frame

# Let's find the top five nearest matches
def reveal_my_twin(x):
    return gl_img.filter_by(x['reference_label'],'id')
spirit_animal = reveal_my_twin(knn_model.query(me_test))
spirit_animal['image'].show()

# Display the top matching image
graphlab.image_analysis.resize(spirit_animal['image'][0:1], 96,96).show()