"""
Framework compatibility layer supporting both legacy Keras 2 / TensorFlow 2.x
and modern Keras 3 / modern TensorFlow.
"""

try:
    import keras

    _keras_version = getattr(keras, "__version__", "2.0.0")
except ImportError:
    try:
        from tensorflow import keras

        _keras_version = getattr(keras, "__version__", "2.0.0")
    except ImportError:
        keras = None
        _keras_version = "0.0.0"


def is_keras_3() -> bool:
    """Return True if running on Keras 3+."""
    if keras is None:
        return False
    try:
        major = int(_keras_version.split(".")[0])
        return major >= 3
    except (ValueError, IndexError):
        return False


# Export core modules
K = keras.backend
models = keras.models
Model = keras.models.Model
Sequential = keras.models.Sequential
load_model = keras.models.load_model

# Optimizers
optimizers = keras.optimizers
Adam = keras.optimizers.Adam

# Callbacks and utils
callbacks = keras.callbacks
ModelCheckpoint = keras.callbacks.ModelCheckpoint
TensorBoard = keras.callbacks.TensorBoard
Callback = keras.callbacks.Callback
plot_model = getattr(keras.utils, "plot_model", None)

# Common layers
layers = keras.layers
Input = keras.layers.Input
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Activation = keras.layers.Activation
LeakyReLU = keras.layers.LeakyReLU
Flatten = keras.layers.Flatten
Concatenate = keras.layers.Concatenate
add = keras.layers.add
Layer = keras.layers.Layer

# 3D Layers
Conv3D = keras.layers.Conv3D
MaxPooling3D = keras.layers.MaxPooling3D
AveragePooling3D = keras.layers.AveragePooling3D
Cropping3D = getattr(keras.layers, "Cropping3D", None)

# 2D Layers
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
AveragePooling2D = keras.layers.AveragePooling2D
ZeroPadding2D = getattr(keras.layers, "ZeroPadding2D", None)

# 1D Layers
Conv1D = keras.layers.Conv1D
MaxPooling1D = keras.layers.MaxPooling1D

# Regularizers
regularizers = keras.regularizers
l2 = keras.regularizers.l2


def create_adam(learning_rate=0.001, **kwargs):
    """Create an Adam optimizer supporting both lr and learning_rate kwargs."""
    if "lr" in kwargs and "learning_rate" not in kwargs:
        kwargs["learning_rate"] = kwargs.pop("lr")
    return Adam(learning_rate=learning_rate, **kwargs)


def save_model_compat(model, filepath, **kwargs):
    """
    Save model safely across Keras 2 and Keras 3.
    Appends appropriate format extension if needed.
    """
    p = str(filepath)
    if is_keras_3():
        if not (p.endswith(".keras") or p.endswith(".h5")):
            p = p + ".keras"
    model.save(p, **kwargs)
    return p
