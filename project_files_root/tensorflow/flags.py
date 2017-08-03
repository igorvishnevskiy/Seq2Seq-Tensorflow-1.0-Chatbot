

import tensorflow as tf
import os


def set_flags():

    tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
    tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                              "Learning rate decays by this much.")
    tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                              "Clip gradients to this norm.")
    tf.app.flags.DEFINE_integer("batch_size", 64,
                                "Batch size to use during training.")
    tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("num_layers", 5, "Number of layers in the model.")
    tf.app.flags.DEFINE_integer("from_vocab_size", 500000, "English vocabulary size.")
    tf.app.flags.DEFINE_integer("to_vocab_size", 500000, "French vocabulary size.")
    tf.app.flags.DEFINE_string("data_dir", "temp_data/", "Data directory")
    tf.app.flags.DEFINE_string("train_dir", "model_out/", "Training directory.")
    tf.app.flags.DEFINE_string("from_train_data", "combined_data/train_input.txt", "Training data.")
    tf.app.flags.DEFINE_string("to_train_data", "combined_data/train_output.txt", "Training data.")
    tf.app.flags.DEFINE_string("from_dev_data", "combined_data/test_input.txt", "Training data.")
    tf.app.flags.DEFINE_string("to_dev_data", "combined_data/test_output.txt", "Training data.")
    tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                                "Limit on the size of training data (0: no limit).")
    tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                                "How many training steps to do per checkpoint.")
    tf.app.flags.DEFINE_boolean("decode", False,
                                "Set to True for interactive decoding.")
    tf.app.flags.DEFINE_boolean("self_test", False,
                                "Run a self-test if this is set to True.")
    tf.app.flags.DEFINE_boolean("use_fp16", False,
                                "Train using fp16 instead of fp32.")

    create_required_directories()


def get_bucket_structure():
    return [(5, 10), (10, 20), (20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (110, 120)]



def get_steps_limit():
    # Training will stop after the limit of steps is reached.
    training_stepa_limit = 100000
    return training_stepa_limit


def save_logs():
    # If you want Tensorflow to save logs, set this to True.
    save_logs_bool = False
    return save_logs_bool


def create_required_directories():
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    if not os.path.exists("model_out/"):
        os.makedirs("model_out/")
    if not os.path.exists("temp_data/"):
        os.makedirs("temp_data/")