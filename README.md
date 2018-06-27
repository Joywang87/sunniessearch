# sunniessearch

I have Built a content recommendation system for sunglasses called “Sunnies Search” by implementing computer vision technology including instance segmentation/object detection and similarity search. 
The 1st step is using instance segmentation model called Mask-RCNN model, which is part of object detection. It is also a classification algorithm on pixel level. The model returns the pixels only containing the sunglasses.
The 2nd step is image similarity search. I extracted the image features using InceptionV3 neural network and then feed it to unsupervised KNN for similar image search in the database. For demo purpose, the database is composed of over 3000 sunglasses items scrapped from Zappos website.

The code for the instance segmentation and similarity search is included. 
