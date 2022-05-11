# Problem definition and specification

Over the last years, deep neural networks have evolved significantly and became quite good at mapping a specific input to the desired output in a supervised learning dataset. In particular, this also applies to the classification of images: with the appropriate training dataset, our models have obtained a very good success rate at classifying a specific image to a specific category. 

However, the models are still not perfect and are not optimized yet in some aspects. In particular, their ability to generalize the knowledge they acquired during training is still not perfect. When applying these algorithms in the real world, the dataset is no longer ideal and might present some inconsistencies with the data that was used to train the network. 

Some examples would include:

- a model that was trained to identify people with daylight images might perform worse when given nighttime images
- a model that was trained to classify flowers indoors might perform worse when given flowers outdoors in a field
- a model that was trained to classify cars from a showroom might perform worse if the cars are on the streets

We would want our models to retain the distinct features of the entities they learned to classify, and apply them when those entities are found in a different environment. 

Elaborating on this idea, we can also observe that classifying an entitiy can be split in multiple smaller subtasks based on the features they present. Breaking up a bigger task into smaller ones is not specifically related to entity classification, but could certainly be applied in that domain as well.

Some additional examples for breaking down tasks:

- a model that detects cars on the road could be trained for detecting motorcycles or buses, because they share the common task of identifying a vehicle
- a model that recognizes and generates speech subtitles in English could be trained for doing the same in German, because they share the common task of recognizing speech
- a model that identifies cats could also be used to identify dogs or other animals that present similar features

This is where Transfer Learning comes in. Transfer Learning is a technique that attempts (but is not limited to) to solve all of the problems above. In this project, we will attempt to use the knowledge learned by an already trained classifier and try to create a model that predicts a totally different classification for the input, which in our case is classifying a flower image into one of 5 categories.