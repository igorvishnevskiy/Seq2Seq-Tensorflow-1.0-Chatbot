
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
api = Api(app)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import logging

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model
import flags


flags.set_flags()


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = flags.get_bucket_structure()



class TF_Talk(Resource):

    def post(self):
        data_received = request.json
        if not data_received:
            data_received = eval(request.form["payload"])

        sentence = data_received["text"]
        print(sentence)

        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        else:
            logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        response = (" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
        print(response)

        return jsonify({"text":response})



def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

try:
    print(sess)
    print("session exist already...")
except Exception, e:
    print("Loading bot model for flask API.")
    sess = tf.Session()
    print("Create model and load parameters.")
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    print("Load vocabularies.")
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
    print("Model has been loaded. Bot is ready to talk to you.")


api.add_resource(TF_Talk, '/tf_talk/')


if __name__ == '__main__':
    app.run(debug=False, port=5000)
