import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ✅ Load and preprocess the images
def load_and_process_image(image_path):
    max_dim = 512  # Resize image to 512 pixels max
    img = Image.open(image_path)
    img = img.resize((max_dim, max_dim))  # Resize
    img = np.array(img, dtype=np.float32)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load images
content_path = "content image.jpg"
style_path = "style image.jpg"

content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)

# ✅ Display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(np.squeeze(content_image.astype('uint8')))
ax1.set_title("Content Image")
ax1.axis("off")

ax2.imshow(np.squeeze(style_image.astype('uint8')))
ax2.set_title("Style Image")
ax2.axis("off")

plt.show()

# ✅ STEP 1: Import Pre-trained VGG19 Model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False  # Freeze model (don't train)

# ✅ Extract Style & Content Features
def get_model():
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    selected_layers = style_layers + content_layers
    outputs = [vgg.get_layer(name).output for name in selected_layers]
    return tf.keras.Model([vgg.input], outputs)

# Load the modified VGG19 model
model = get_model()

# ✅ Extract features from an image
def get_features(image, model):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    outputs = model(image)
    style_outputs = outputs[:5]
    content_outputs = outputs[5:]
    return {'style': style_outputs, 'content': content_outputs}

# Extract features for both images
content_features = get_features(content_image, model)
style_features = get_features(style_image, model)

# ✅ Compute Content Loss
def compute_content_loss(content_feature, generated_feature):
    return tf.reduce_mean(tf.square(content_feature - generated_feature))

# ✅ Compute Gram Matrix (for Style Loss)
def gram_matrix(feature_maps):
    channels = int(feature_maps.shape[-1])
    features = tf.reshape(feature_maps, [-1, channels])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram

# ✅ Compute Style Loss
def compute_style_loss(style_features, generated_features):
    style_loss = 0
    for sf, gf in zip(style_features, generated_features):
        gram_sf = gram_matrix(sf)
        gram_gf = gram_matrix(gf)
        style_loss += tf.reduce_mean(tf.square(gram_sf - gram_gf))
    return style_loss

# ✅ Compute Total Loss
def compute_total_loss(content_features, style_features, generated_features, alpha=1e4, beta=1e-2):
    content_loss = compute_content_loss(content_features['content'][0], generated_features['content'][0])
    style_loss = compute_style_loss(style_features['style'], generated_features['style'])
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

# ✅ Initialize generated image
generated_image = tf.Variable(content_image, dtype=tf.float32)

# ✅ Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

# ✅ Training function
@tf.function
def train_step(generated_image, content_features, style_features, model):
    with tf.GradientTape() as tape:
        generated_features = get_features(generated_image, model)
        loss = compute_total_loss(content_features, style_features, generated_features)
    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, 0, 255))
    return loss

# ✅ Run optimization
epochs = 1000
display_interval = 100

for epoch in range(epochs):
    loss = train_step(generated_image, content_features, style_features, model)
    if epoch % display_interval == 0:
        print(f"Iteration {epoch}: Loss = {loss.numpy()}")
        output_image = np.array(generated_image.numpy(), dtype=np.uint8)
        plt.imshow(np.squeeze(output_image))
        plt.axis("off")
        plt.show()

# ✅ Save final image
final_image = Image.fromarray(np.squeeze(output_image))
final_image.save("stylized_output.jpg")
print("🎉 Stylized image saved as stylized_output.jpg")