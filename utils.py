import tensorflow as tf
import os
from Data_Process import CSVDataPreprocessor
from Data_Process import PickleDataPreprocessor


def save_model(model, path, format='pb'):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if format == 'pb':
        model.save(path)
        print(f"✅  Model saved in TensorFlow SavedModel format to {path}/")
    elif format == 'h5':
        print("⚠️  Subclassed models cannot be saved in .h5 format. Switching to .pb format.")
        tf.saved_model.save(model, path)
    elif format == 'weights':
        if not path.endswith('.h5'):
            path += '.h5'
        model.save_weights(path)
        print(f"✅  Weights saved to {path}")
    else:
        raise ValueError(f"Unsupported save_format: {format}")


def load_model(path):
    model = tf.keras.models.load_model(path)
    print(f"✅  Model loaded from {path}")
    return model


def get_data_preprocessor(path, time_steps=5):
    """
    Based on file extension, return the corresponding data preprocessor.
    :param path: data file path
    Support: .csv, .pkl, .p, .pickle
    """
    ext = os.path.splitext(path)[-1].lower()
    
    if ext == '.csv':
        return CSVDataPreprocessor(path, time_steps)
    
    elif ext in ('.pkl', '.p', '.pickle'):
        return PickleDataPreprocessor(path, time_steps)
    
    else:
        raise ValueError(f"Unsupported file extension {ext}, please use .csv or .pkl/.p/.pickle")