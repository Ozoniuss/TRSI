# Problem analysis and solution

In this section we will analyze some of the potential benefits and concerns within our problem and some considerations about our solution.

## Potential concerns

Obviously, the main concern with transfer learning is why would we even expect that the "knowledge" gathered by a model when performing a specific task could be transfered to another task. After all, we can't expect a machine learning model to break down the distinct features of an entity like we do. 

For example, when we want to classify an entity as human, we would be looking for some distinct features like arms, legs, head, hair, skin color etc., or basically any other characteristic that a person has. On the other hand, a machine learning model might identify a completely different set of characteristics they found in the input pixels that has nothing to do with the way our brains perform this type of classification. This could potentially limit their ability of applying certain patterns they already learned to perform a different but related classification, because the entities in the new task might not present those same patterns. Of course, the opposite might also be true; there is no guarantee that the way our brain breaks down a task is the best one.

## Potential benefits

Now let's examine if there are benefits of using Transfer Learning over simply including more images in our training dataset in the first place. The main benefits this approach could provide are the following:

- `Better initial model` In other types of learning, the model is build without any knowledge and the weights and biases are initialized randomly. Transfer learning could offer a better starting point and can perform tasks at some level without even training.
- `Higher learning rate` Transfer learning could offer a higher learning rate during training since the problem has already trained for a similar task.
- `Faster training` The learning can achieve the desired performance faster than traditional learning methods since it leverages a pre-trained model.

However, the final performance of transfer learning might not be much higher than traditional learning models, if even higher. Though where this approach could excel is saving resources. Since the core of the main model had already been trained, only a small subsets of weights and biases are adjusted in the training process. This could significantly reduce resource usage and would be very beneficial in case training resources (in particular time, memory and the dataset) are limited. 

We will be evaluating the performance of a Transfer Learning solution to performing a classification task. We will be attaching 5 output neurons to the [MobileNet v2 Feature Vector](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) in order to train the model to classify flowers in one of the following 5 categories: roses, daisies, dandelions, sunflowers, tulips. The flower dataset used for training can be downloaded [at this link.](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)