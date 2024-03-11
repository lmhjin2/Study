from keras_cv_attention_models import efficientvit, test_images
model = efficientvit.EfficientViT_M1(input_shape=(256, 256, 3))

# Run prediction
preds = model(model.preprocess_input(test_images.cat()))
print(model.decode_predictions(preds))
# [('n02124075', 'Egyptian_cat', 0.50921583), ('n02123045', 'tabby', 0.14553155), ...]