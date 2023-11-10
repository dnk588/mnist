from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_NAME = 'model_fmr_all.h5'
model = load_model(MODEL_NAME)


def process(image_file):
    picture = Image.open(image_file).convert('L').resize((28, 28))
    new_pic = image.img_to_array(picture)
    new_pic = new_pic.reshape(-1, 28, 28)
    prediction = model.predict(new_pic)

    return prediction
