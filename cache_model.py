# Download and uncompress the model. See also:
# https://www.tensorflow.org/hub/tf2_saved_model#using_a_savedmodel_in_low-level_tensorflow
# https://www.tensorflow.org/hub/caching

import tensorflow_hub as hub

TF_HUB_MODEL_PATH = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"

hub.load(TF_HUB_MODEL_PATH)
print(hub.resolve(TF_HUB_MODEL_PATH))