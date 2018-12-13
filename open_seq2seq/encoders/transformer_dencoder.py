# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from open_seq2seq.parts.transformer import utils, attention_layer, \
                                           embedding_layer, ffn_layer, beam_search
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                                  LayerNormalization
from .encoder import Encoder


class TransformerDENcoder(Encoder):
  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_required_params(), **{
        'layer_postprocess_dropout': float,
        'num_hidden_layers': int,
        'hidden_size': int,
        'num_heads': int,
        'attention_dropout': float,
        'relu_dropout': float,
        'filter_size': int,
        'src_vocab_size': int,
    })

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_optional_params(), **{
        'regularizer': None,  # any valid TensorFlow regularizer
        'regularizer_params': dict,
        'initializer': None,  # any valid TensorFlow initializer
        'initializer_params': dict,
        'pad_embeddings_2_eight': bool,
    })

  def _cast_types(self, input_dict):
    return input_dict

  def __init__(self, params, model,
               name="transformer_DENcoder", mode='train'):
    super(TransformerDENcoder, self).__init__(params, model, name, mode)
    self.embedding_softmax_layer = None
    self.output_normalization = None
    self._mode = mode
    self.layers = []
    # in original T paper embeddings are shared between encoder and decoder
    # also final projection = transpose(E_weights), we currently only support
    # this behaviour
    self.params['shared_embed'] = True

  def _encode(self, input_dict):
    with tf.name_scope("DENcode"):
      # prepare decoder layers
      if len(self.layers) == 0:
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
          self.params["src_vocab_size"], self.params["hidden_size"],
          pad_vocab_to_eight=self.params.get('pad_embeddings_2_eight', False),
        )

        for _ in range(self.params["num_hidden_layers"]):
          self_attention_layer = attention_layer.SelfAttention(
              self.params["hidden_size"], self.params["num_heads"],
              self.params["attention_dropout"],
              self.mode == "train",
          )
          enc_dec_attention_layer = attention_layer.Attention(
              self.params["hidden_size"], self.params["num_heads"],
              self.params["attention_dropout"],
              self.mode == "train",
          )
          feed_forward_network = ffn_layer.FeedFowardNetwork(
              self.params["hidden_size"], self.params["filter_size"],
              self.params["relu_dropout"], self.mode == "train",
          )

          self.layers.append([
              PrePostProcessingWrapper(self_attention_layer, self.params,
                                       self.mode == "train"),
              PrePostProcessingWrapper(enc_dec_attention_layer, self.params,
                                       self.mode == "train"),
              PrePostProcessingWrapper(feed_forward_network, self.params,
                                       self.mode == "train")
          ])

        self.output_normalization = LayerNormalization(
            self.params["hidden_size"]
        )

      #if targets is None:
      #  return self.predict(encoder_outputs, inputs_attention_bias)
      #else:
      inputs = input_dict['source_tensors'][0]
      inputs_attention_bias = utils.get_padding_bias(
        inputs)
      logits = self.decode_pass(inputs,
                                self.embedding_softmax_layer(inputs),
                                inputs_attention_bias)

      #return {"logits": logits,
      #        "outputs": [tf.argmax(logits, axis=-1)],
      #        "final_state": None,
      #        "final_sequence_lengths": None}
      return {'outputs': logits,
              'inputs_attention_bias': inputs_attention_bias,
              'state': None,
              'src_lengths': input_dict['source_tensors'][1],
              'embedding_softmax_layer': self.embedding_softmax_layer,
              'encoder_input': inputs}


  def _call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
            attention_bias, cache=None):
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          # TODO: Figure out why this is needed
          # decoder_self_attention_bias = tf.cast(x=decoder_self_attention_bias,
          #                                      dtype=decoder_inputs.dtype)
          decoder_inputs = self_attention_layer(
              decoder_inputs, decoder_self_attention_bias, cache=layer_cache,
          )
        with tf.variable_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs, encoder_outputs, attention_bias,
          )
        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    return self.output_normalization(decoder_inputs)

  def decode_pass(self, targets, encoder_outputs, inputs_attention_bias):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
      inputs_attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    # Prepare inputs to decoder layers by shifting targets, adding positional
    # encoding and applying dropout.
    decoder_inputs = self.embedding_softmax_layer(targets)
    #with tf.name_scope("shift_targets"):
    #  # Shift targets to the right, and remove the last element
    #  decoder_inputs = tf.pad(
    #      decoder_inputs, [[0, 0], [1, 0], [0, 0]],
    #  )[:, :-1, :]
    with tf.name_scope("add_pos_encoding"):
      length = tf.shape(decoder_inputs)[1]
      decoder_inputs += tf.cast(
          utils.get_position_encoding(length, self.params["hidden_size"]),
          dtype=self.params['dtype'],
      )
    if self.mode == "train":
      decoder_inputs = tf.nn.dropout(
          decoder_inputs, 1 - self.params["layer_postprocess_dropout"],
      )

    # Run values
    decoder_self_attention_bias = utils.get_decoder_self_attention_bias(length)
    #decoder_self_attention_bias = tf.zeros_like(decoder_self_attention_bias)

    # do decode
    outputs = self._call(
        decoder_inputs=decoder_inputs,
        encoder_outputs=encoder_outputs,
        decoder_self_attention_bias=decoder_self_attention_bias,
        attention_bias=inputs_attention_bias,
    )

    logits = self.embedding_softmax_layer.linear(outputs)
    return logits
