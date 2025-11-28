import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

MODEL_PATH = "models/mnist_model.h5"


# Load your saved model
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))

    img_arr = np.array(img)

    # OPTIONAL:
    # If your digits are dark writing on a white background:
    # uncomment this line:
    # img_arr = 255 - img_arr

    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 28 * 28)
    return img_arr

def predict_digit(image_path):
    x = preprocess_image(image_path)
    preds = model.predict(x)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    return predicted_class, preds[0]

def show_preprocessed(image_path):
    x = preprocess_image(image_path)
    img_28 = x.reshape(28, 28)
    plt.imshow(img_28, cmap="gray")
    plt.title("Preprocessed 28x28 image")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    img_path = "examples/example_digit_black_6.png"
    show_preprocessed(img_path)
    digit, probs = predict_digit(img_path)
    print("Predicted digit:", digit)
    print("Probabilities:", probs)
