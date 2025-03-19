import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# ✅ Load and preprocess the images
def load_and_process_image(image_path):
    max_dim = 512
    img = Image.open(image_path).convert('RGB')  # Convert to RGB to avoid issues
    img = img.resize((max_dim, max_dim), Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.vgg19.preprocess_input(img)  # Normalize for VGG19
    return img

# ✅ Display helper function
def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))  # Remove batch dimension
    img = img + [123.68, 116.779, 103.939]  # Reverse normalization
    img = np.clip(img, 0, 255).astype("uint8")  # Clip values to valid range
    return img

# ✅ Import VGG19 Model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False  # Freeze model
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    selected_layers = style_layers + content_layers
    outputs = [vgg.get_layer(name).output for name in selected_layers]
    return tf.keras.Model([vgg.input], outputs)

# ✅ Extract features
def get_features(image, model):
    outputs = model(image)
    style_outputs = outputs[:5]
    content_outputs = outputs[5:]
    return {'style': style_outputs, 'content': content_outputs}

# ✅ Compute Content Loss
def compute_content_loss(content_feature, generated_feature):
    return tf.reduce_mean(tf.square(content_feature - generated_feature))

# ✅ Compute Gram Matrix (for Style Loss)
def gram_matrix(feature_maps):
    channels = int(feature_maps.shape[-1])
    features = tf.reshape(feature_maps, [-1, channels])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(tf.shape(features)[0], tf.float32)

# ✅ Compute Style Loss
def compute_style_loss(style_features, generated_features):
    style_loss = 0
    style_layer_weights = [0.5, 1.0, 1.5, 3.0, 4.0]  # Tuned weights
    for sf, gf, weight in zip(style_features, generated_features, style_layer_weights):
        gram_sf = gram_matrix(sf)
        gram_gf = gram_matrix(gf)
        style_loss += weight * tf.reduce_mean(tf.square(gram_sf - gram_gf))
    return style_loss / len(style_features)

# ✅ Compute Total Loss
def compute_total_loss(content_features, style_features, generated_features, alpha=1e4, beta=1e3):
    content_loss = compute_content_loss(content_features['content'][0], generated_features['content'][0])
    style_loss = compute_style_loss(style_features['style'], generated_features['style'])
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

# ✅ Optimization
def run_style_transfer(content_path, style_path, epochs=500, lr=5e-4):
    # Load images
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    # Load model
    model = get_model()

    # Extract features
    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)

    # Initialize generated image
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Training function
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            generated_features = get_features(generated_image, model)
            loss = compute_total_loss(content_features, style_features, generated_features)
        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0, 255))
        return loss

    # Training loop
    for epoch in range(epochs):
        loss = train_step()
        if epoch % 100 == 0:  # Show updates every 100 iterations
            print(f"Iteration {epoch}: Loss = {loss.numpy()}")

    # Save final output
    output_image = deprocess_image(generated_image.numpy())
    final_image = Image.fromarray(output_image)
    final_image.save("stylized_output.jpg")
    
    return final_image

# ✅ Gradio Interface
def nst_interface(content_image, style_image):
    content_image.save("content_temp.jpg")
    style_image.save("style_temp.jpg")
    result = run_style_transfer("content_temp.jpg", "style_temp.jpg")
    return result

# ✅ Launch Gradio UI
interface = gr.Interface(
    fn=nst_interface,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil")],
    outputs=gr.Image(type="pil"),
    title="Neural Style Transfer",
    description="Upload a content image and a style image to generate a stylized image.",
)

if __name__ == "__main__":
    interface.launch()
