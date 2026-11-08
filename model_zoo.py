"""Six classification models taken from the architectures in the reference papers.

  custom_cnn       - your synopsis CNN (Conv32-64-128 -> Dense256 -> Dropout -> Dense128)
  vgg16            - VGG16 transfer learning            (Rozlan & Hanafi 2022)
  inception_v3     - InceptionV3 transfer learning      (Rozlan & Hanafi 2022)
  efficientnet_b4  - EfficientNetB4 transfer learning   (Pratap & Kumar 2023)
  mobilenet_v2     - MobileNetV2 (light / edge model)   (Hamim & Jony 2024)
  vit              - small Vision Transformer, 16x16 patches (image encoder of Annaram & Gopichand 2026)

Importing this module also registers the custom ViT layer so saved models can be reloaded.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import applications as A
import config as C

IMG_SHAPE = (*C.IMG_SIZE, 3)


# ---------------------------------------------------------------- helpers
def _augment(x):
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    return x


def _base(fn, **kw):
    try:
        return fn(include_top=False, weights="imagenet", input_shape=IMG_SHAPE, **kw)
    except Exception as e:  # offline etc.
        print(f"[WARNING] could not load ImageNet weights ({e}). Training from scratch - accuracy will be lower.")
        return fn(include_top=False, weights=None, input_shape=IMG_SHAPE, **kw)


def _transfer(name, fn, pre, num_classes):
    inputs = layers.Input(shape=IMG_SHAPE)
    x = _augment(inputs)
    if pre is not None:
        x = pre(x)
    base = _base(fn)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, out, name=name)


def get_base(model):
    """Return the pretrained backbone sub-model (None for custom_cnn / vit)."""
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            return l
    return None


# ---------------------------------------------------------------- ViT
@tf.keras.utils.register_keras_serializable(package="chilli")
class AddPositionEmbedding(layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, input_shape[1], input_shape[2]),
                                   initializer="random_normal", trainable=True)

    def call(self, x):
        return x + self.pos


def build_vit(num_classes, patch=16, dim=96, depth=4, heads=4, mlp=192):
    inputs = layers.Input(shape=IMG_SHAPE)
    x = _augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(dim, patch, strides=patch)(x)                 # patch embedding
    n = (C.IMG_SIZE[0] // patch) * (C.IMG_SIZE[1] // patch)
    x = layers.Reshape((n, dim))(x)
    x = AddPositionEmbedding()(x)
    for _ in range(depth):
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=0.1)(h, h)
        x = layers.Add()([x, h])
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.Dense(mlp, activation="gelu")(h)
        h = layers.Dropout(0.1)(h)
        h = layers.Dense(dim)(h)
        x = layers.Add()([x, h])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, out, name="vit")


# ---------------------------------------------------------------- builders
def build_custom_cnn(n):
    from train import build_model
    return build_model(n)


def build_vgg16(n):
    pre = layers.Normalization(mean=[123.68, 116.779, 103.939], variance=[1.0, 1.0, 1.0])
    return _transfer("vgg16", A.VGG16, pre, n)


def build_inception_v3(n):
    return _transfer("inception_v3", A.InceptionV3, layers.Rescaling(1 / 127.5, offset=-1), n)


def build_efficientnet_b4(n):
    return _transfer("efficientnet_b4", A.EfficientNetB4, None, n)   # has built-in preprocessing (0-255 input)


def build_mobilenet_v2(n):
    return _transfer("mobilenet_v2", A.MobileNetV2, layers.Rescaling(1 / 127.5, offset=-1), n)


MODELS = {
    "custom_cnn": build_custom_cnn,
    "vgg16": build_vgg16,
    "inception_v3": build_inception_v3,
    "efficientnet_b4": build_efficientnet_b4,
    "mobilenet_v2": build_mobilenet_v2,
    "vit": build_vit,
}

DESCRIPTIONS = {
    "custom_cnn": "Custom CNN (synopsis)",
    "vgg16": "VGG16 (transfer)",
    "inception_v3": "InceptionV3 (transfer)",
    "efficientnet_b4": "EfficientNetB4 (transfer)",
    "mobilenet_v2": "MobileNetV2 (transfer)",
    "vit": "Vision Transformer (scratch)",
}


def make_optimizer(name, lr):
    if name == "vit":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    return tf.keras.optimizers.Adam(learning_rate=lr)
