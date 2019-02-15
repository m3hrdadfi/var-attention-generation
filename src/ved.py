from nltk.tokenize import WhitespaceTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
import os
import pickle
import time
from tqdm import tqdm
from .attention import (
    LuongAttention,
    AttentionWrapper
)
from .decoder import dynamic_decode
from .basic_decoder import BasicDecoder
from .preparation import get_batches
from .evaluation import calculate_bleu_scores, calculate_entropy, calculate_ngram_diversity


class VarSeq2SeqVarAttnModel(object):

    def __init__(self,
                 config,
                 encoder_embeddings_matrix,
                 decoder_embeddings_matrix,
                 encoder_word_index,
                 decoder_word_index):

        self.config = config

        self.lstm_hidden_units = config['lstm_hidden_units']
        self.embedding_size = config['embedding_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']

        self.encoder_vocab_size = config['encoder_vocab']
        self.decoder_vocab_size = config['decoder_vocab']

        self.encoder_num_tokens = config['encoder_num_tokens']
        self.decoder_num_tokens = config['decoder_num_tokens']

        self.dropout_keep_prob = config['dropout_keep_prob']
        self.word_dropout_keep_probability = config['word_dropout_keep_probability']
        self.distribution = config['distribution']
        self.z_temp = config['z_temp']
        self.attention_temp = config['attention_temp']
        self.use_hmean = config['use_hmean']
        self.gamma_val = config['gamma_val']

        self.initial_learning_rate = config['initial_learning_rate']
        self.learning_rate_decay = config['learning_rate_decay']
        self.min_learning_rate = config['min_learning_rate']

        self.batch_size = config['batch_size']
        self.epochs = config['n_epochs']

        self.encoder_embeddings_matrix = encoder_embeddings_matrix
        self.decoder_embeddings_matrix = decoder_embeddings_matrix
        self.encoder_word_index = encoder_word_index
        self.decoder_word_index = decoder_word_index
        self.encoder_idx_word = dict((i, word) for word, i in encoder_word_index.items())
        self.decoder_idx_word = dict((i, word) for word, i in decoder_word_index.items())

        self.logs_dir = config['logs_dir']
        self.model_checkpoint_dir = config['model_checkpoint_dir']
        self.bleu_path = config['bleu_path']

        self.pad = self.decoder_word_index['PAD']
        self.eos = self.decoder_word_index['EOS']

        self.epoch_bleu_score_val = {'1': [], '2': [], '3': [], '4': []}
        self.log_str = []
        self.log_str_dir = config['log_str_dir']
        self.wrapper = config['wrapper']

        if self.logs_dir:
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)

        if self.log_str_dir:
            if not os.path.exists(self.log_str_dir):
                os.makedirs(self.log_str_dir)

        if self.bleu_path:
            if not os.path.exists(os.path.dirname(self.bleu_path)):
                os.makedirs(os.path.dirname(self.bleu_path))

        self.build_model()

    def build_model(self):
        print('[INFO] Building models ...')

        self.init_placeholders()
        self.embedding_layer()
        self.build_encoder()
        self.build_latent_space()
        self.build_decoder()
        self.loss()
        self.optimize()
        self.summary()

    def init_placeholders(self):
        print('----> [INFO] Create placeholders for inputs to the model.')

        with tf.name_scope('model_inputs'):
            # Create palce holders for inputs to the model
            self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.encoder_num_tokens], name='input')
            self.target_data = tf.placeholder(tf.int32, [self.batch_size, self.decoder_num_tokens], name='targets')
            self.lr = tf.placeholder(tf.float32, name='learning_rate', shape=())
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # Dropout Keep Probability
            self.source_sentence_length = tf.placeholder(
                tf.int32, shape=(self.batch_size,),
                name='source_sentence_length')
            self.target_sentence_length = tf.placeholder(
                tf.int32, shape=(self.batch_size,),
                name='target_sentence_length')
            self.word_dropout_keep_prob = tf.placeholder(tf.float32, name='word_drop_keep_prob', shape=())
            self.lambda_coeff = tf.placeholder(tf.float32, name='lambda_coeff', shape=())
            self.gamma_coeff = tf.placeholder(tf.float32, name='gamma_coeff', shape=())
            self.z_temperature = tf.placeholder(tf.float32, name='z_temperature', shape=())
            self.attention_temperature = tf.placeholder(tf.float32, name='attention_temperature', shape=())

    def embedding_layer(self):
        print('----> [INFO] Create word embedding for encoder and decoder parts.')

        with tf.name_scope('word_embeddings'):
            with tf.name_scope('wes_encoder'):
                self.encoder_embeddings = tf.Variable(
                    initial_value=np.array(self.encoder_embeddings_matrix, dtype=np.float32),
                    dtype=tf.float32, trainable=False)
                self.enc_embed_input = tf.nn.embedding_lookup(self.encoder_embeddings, self.input_data)
                # self.enc_embed_input = tf.nn.dropout(self.enc_embed_input, keep_prob=self.keep_prob)

            with tf.name_scope('wes_decoder'):
                self.decoder_embeddings = tf.Variable(
                    initial_value=np.array(self.decoder_embeddings_matrix, dtype=np.float32),
                    dtype=tf.float32, trainable=False)

                keep = tf.where(
                    tf.random_uniform([self.batch_size, self.decoder_num_tokens]) < self.word_dropout_keep_prob,
                    tf.fill([self.batch_size, self.decoder_num_tokens], True),
                    tf.fill([self.batch_size, self.decoder_num_tokens], False))
                ending = tf.cast(keep, dtype=tf.int32) * self.target_data

                # Minus 1 implies everything till the last dim
                ending = tf.strided_slice(ending, [0, 0], [self.batch_size, -1], [1, 1], name='slice_input')

                self.dec_input = tf.concat([
                    tf.fill([self.batch_size, 1], self.decoder_word_index['GO']), ending], 1, name='dec_input')
                self.dec_embed_input = tf.nn.embedding_lookup(self.decoder_embeddings, self.dec_input)
                # self.dec_embed_input = tf.nn.dropout(self.dec_embed_input, keep_prob=self.keep_prob)

    def build_encoder(self):
        print('----> [INFO] Create encoder block.')

        with tf.name_scope('encode'):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer + 1)):
                    # cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob)

                    # cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_units)
                    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob)

                    self.enc_output, self.enc_state = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw,
                        cell_bw,
                        self.enc_embed_input,
                        self.source_sentence_length,
                        dtype=tf.float32)

            # Join outputs since we are using a bidirectional RNN
            # Concatenated h from the fw and bw LSTMs
            self.h_N = tf.concat([self.enc_state[0][1], self.enc_state[1][1]], axis=-1, name='h_N')
            self.enc_outputs = tf.concat([self.enc_output[0], self.enc_output[1]], axis=-1, name='encoder_outputs')

    def build_latent_space(self):
        print('----> [INFO] Create latent space.')
        with tf.name_scope('latent_space'):
            if self.distribution == 'dirichlet':
                self.z_alpha = Dense(self.latent_dim, activation=tf.nn.tanh, name='z_alpha')(self.h_N)
                self.z_scale = Dense(self.latent_dim, activation=tf.nn.softplus, name='z_scale')(self.h_N)
                self.z_log_alpha = tf.exp(self.z_scale * self.z_alpha, name='z_log_alpha')

                self.z_mean = tf.add(
                    tf.digamma(self.z_log_alpha[:, :self.latent_dim - 1]),
                    - tf.digamma(self.z_log_alpha[:, -1:]),
                    name='z_mean'
                )
                _one = tf.cast(1, dtype=self.z_log_alpha.dtype)
                self.z_log_sigma = tf.sqrt(
                    tf.add(
                        tf.polygamma(_one, self.z_log_alpha[:, :self.latent_dim - 1]),
                        tf.polygamma(_one, self.z_log_alpha[:, -1:])
                    ),
                    name='z_log_sigma'
                )
                self.z_vector = tf.identity(self.sample(), name='z_vector')
            else:
                self.z_mean = Dense(self.latent_dim, name='z_mean')(self.h_N)
                self.z_log_sigma = Dense(self.latent_dim, name='z_log_sigma')(self.h_N)

                self.z_vector = tf.identity(self.sample(), name='z_vector')

    def build_decoder(self):
        print('----> [INFO] Create decoder block.')

        with tf.variable_scope('decode'):
            for layer in range(self.num_layers):
                with tf.variable_scope('decoder_{}'.format(layer + 1)):
                    dec_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2 * self.lstm_hidden_units)
                    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=self.keep_prob)

            self.output_layer = Dense(self.decoder_vocab_size)

            attn_mech = LuongAttention(
                2 * self.lstm_hidden_units,
                self.enc_outputs,
                memory_sequence_length=self.source_sentence_length)

            attn_cell = AttentionWrapper(
                dec_cell,
                attn_mech,
                self.attention_temperature,
                self.use_hmean,
                self.lstm_hidden_units)

            self.init_state = attn_cell.zero_state(self.batch_size, tf.float32)

            with tf.name_scope('training_decoder'):
                training_helper = seq2seq.TrainingHelper(
                    inputs=self.dec_embed_input,
                    sequence_length=self.target_sentence_length,
                    time_major=False)

                training_decoder = BasicDecoder(
                    attn_cell,
                    training_helper,
                    initial_state=self.init_state,
                    latent_vector=self.z_vector,
                    output_layer=self.output_layer)

                self.training_logits, _state, _len, self.c_kl_batch_train = dynamic_decode(
                    training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.decoder_num_tokens)

                self.training_logits = tf.identity(self.training_logits.rnn_output, 'logits')

            with tf.name_scope('inference_decoder'):
                start_token = self.decoder_word_index['GO']
                end_token = self.decoder_word_index['EOS']

                start_tokens = tf.tile(
                    tf.constant([start_token], dtype=tf.int32),
                    [self.batch_size],
                    name='start_tokens')

                inference_helper = seq2seq.SampleEmbeddingHelper(
                    self.decoder_embeddings,
                    start_tokens,
                    end_token)

                # inference_helper = seq2seq.GreedyEmbeddingHelper(
                #     self.decoder_embeddings,
                #     start_tokens,
                #     end_token)

                inference_decoder = BasicDecoder(
                    attn_cell,
                    inference_helper,
                    initial_state=self.init_state,
                    latent_vector=self.z_vector,
                    output_layer=self.output_layer)

                self.inference_logits, _state, _len, self.c_kl_batch_inf = dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.decoder_num_tokens)

                self.inference_logits = tf.identity(self.inference_logits.sample_id, name='predictions')

                # Divide by respective target seq lengths
                self.c_kl_batch_train = tf.div(
                    self.c_kl_batch_train,
                    tf.cast(self.target_sentence_length, dtype=tf.float32))

    def loss(self):
        with tf.name_scope('losses'):
            self.kl_loss = self.calculate_kl_loss()
            self.kl_loss = tf.scalar_mul(self.lambda_coeff, self.kl_loss)

            self.context_kl_loss = tf.scalar_mul(self.gamma_coeff * self.lambda_coeff, self.c_kl_batch_train)

            batch_maxlen = tf.reduce_max(self.target_sentence_length)

            # the training decoder only emits outputs equal in time-steps to the
            # max time in the current batch
            target_sequence = tf.slice(
                input_=self.target_data,
                begin=[0, 0],
                size=[self.batch_size, batch_maxlen],
                name='target_sequence')

            # Create the weights for sequence_loss
            masks = tf.sequence_mask(self.target_sentence_length, batch_maxlen, dtype=tf.float32, name='masks')

            self.xent_loss = seq2seq.sequence_loss(
                self.training_logits,
                target_sequence,
                weights=masks,
                average_across_batch=False)

            # L2-Regularization
            self.var_list = tf.trainable_variables()
            self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list if 'bias' not in v.name]) * 0.001

            self.cost = tf.reduce_sum(self.xent_loss + self.kl_loss + self.context_kl_loss) + self.lossL2

    def optimize(self):
        # Optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost, var_list=self.var_list)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('xent_loss', tf.reduce_sum(self.xent_loss))
            tf.summary.scalar('l2_loss', tf.reduce_sum(self.lossL2))
            tf.summary.scalar("kl_loss", tf.reduce_sum(self.kl_loss))
            tf.summary.scalar('context_kl_loss', tf.reduce_sum(self.context_kl_loss))
            tf.summary.scalar('total_loss', tf.reduce_sum(self.cost))
            tf.summary.histogram('latent_vector', self.z_vector)
            tf.summary.histogram('latent_mean', self.z_mean)
            tf.summary.histogram('latent_log_sigma', self.z_log_sigma)

            if self.distribution == 'dirichlet':
                tf.summary.histogram('latent_log_alpha', self.z_log_alpha)
                tf.summary.histogram('latent_alpha', self.z_alpha)
                tf.summary.histogram('latent_scale', self.z_scale)

            self.summary_op = tf.summary.merge_all()

    def train(self, x_train, y_train, x_val, y_val, true_val):
        print('[INFO] Training process started ...')

        learning_rate = self.initial_learning_rate
        iter_i = 0
        lambda_val = 0.0

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(self.logs_dir + '/train', sess.graph)

            for epoch_i in range(1, self.epochs + 1):

                start_time = time.time()
                for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                        get_batches(x_train, y_train, self.batch_size)):

                    try:
                        iter_i += 1

                        _, _summary = sess.run(
                            [self.train_op, self.summary_op],
                            feed_dict={
                                self.input_data: input_batch,
                                self.target_data: output_batch,
                                self.lr: learning_rate,
                                self.source_sentence_length: source_sent_lengths,
                                self.target_sentence_length: tar_sent_lengths,
                                self.keep_prob: self.dropout_keep_prob,
                                self.lambda_coeff: lambda_val,
                                self.z_temperature: self.z_temp,
                                self.word_dropout_keep_prob: self.word_dropout_keep_probability,
                                self.attention_temperature: self.attention_temp,
                                self.gamma_coeff: self.gamma_val
                            })

                        writer.add_summary(_summary, iter_i)

                        # KL Annealing till some iteration
                        if iter_i <= 3000:
                            lambda_val = np.round((np.tanh((iter_i - 4500) / 1000) + 1) / 2, decimals=6)

                    except Exception as e:
                        # print(iter_i, e)
                        pass

                self.validate(sess, x_val, y_val, true_val)
                val_bleu_str = str(self.epoch_bleu_score_val['1'][epoch_i - 1]) + ' | ' \
                               + str(self.epoch_bleu_score_val['2'][epoch_i - 1]) + ' | ' \
                               + str(self.epoch_bleu_score_val['3'][epoch_i - 1]) + ' | ' \
                               + str(self.epoch_bleu_score_val['4'][epoch_i - 1])

                # Reduce learning rate, but not below its minimum value
                learning_rate = np.max([self.min_learning_rate, learning_rate * self.learning_rate_decay])

                saver = tf.train.Saver()
                saver.save(sess, self.model_checkpoint_dir + str(epoch_i) + ".ckpt")
                end_time = time.time()

                # Save the validation BLEU scores so far
                with open(self.bleu_path + '.pkl', 'wb') as f:
                    pickle.dump(self.epoch_bleu_score_val, f)

                self.log_str.append('Epoch {:>3}/{} - Time {:>6.1f} BLEU: {}'.format(
                    epoch_i,
                    self.epochs,
                    end_time - start_time,
                    val_bleu_str))

                with open('%s/%s' % (self.log_str_dir, 'logs.txt'), 'w') as f:
                    f.write('\n'.join(self.log_str))

                print(self.log_str[-1])

    def validate(self, sess, x_val, y_val, true_val):
        # Calculate BLEU on validation data
        hypotheses_val = []
        references_val = []
        symbol = []

        if self.config['experiment'] == 'qgen':
            symbol.append('?')

        for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                get_batches(x_val, y_val, self.batch_size)):

            answer_logits = sess.run(self.inference_logits, feed_dict={
                self.input_data: input_batch,
                self.source_sentence_length: source_sent_lengths,
                self.keep_prob: 1.0,
                self.word_dropout_keep_prob: 1.0,
                self.z_temperature: self.z_temp,
                self.attention_temperature: self.attention_temp})

            for k, pred in enumerate(answer_logits):
                hypotheses_val.append(
                    WhitespaceTokenizer().tokenize(
                        " ".join(
                            [self.decoder_idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])) + symbol)
                references_val.append([WhitespaceTokenizer().tokenize(true_val[batch_i * self.batch_size + k])])

        bleu_scores = calculate_bleu_scores(references_val, hypotheses_val)
        self.epoch_bleu_score_val['1'].append(bleu_scores[0])
        self.epoch_bleu_score_val['2'].append(bleu_scores[1])
        self.epoch_bleu_score_val['3'].append(bleu_scores[2])
        self.epoch_bleu_score_val['4'].append(bleu_scores[3])

    def predict(self, checkpoint, x_test, y_test, true_test):
        pred_logits = []
        hypotheses_test = []
        references_test = []
        symbol = []
        if self.config['experiment'] == 'qgen':
            symbol.append('?')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                    get_batches(x_test, y_test, self.batch_size)):

                result = sess.run(self.inference_logits, feed_dict={
                    self.input_data: input_batch,
                    self.source_sentence_length: source_sent_lengths,
                    self.keep_prob: 1.0,
                    self.word_dropout_keep_prob: 1.0,
                    self.z_temperature: self.z_temp,
                    self.attention_temperature: self.attention_temp})

                pred_logits.extend(result)

                for k, pred in enumerate(result):
                    hypotheses_test.append(
                        WhitespaceTokenizer().tokenize(" ".join(
                            [self.decoder_idx_word[i] for i in pred if i not in [self.pad, -1, self.eos]])) + symbol)
                    references_test.append([WhitespaceTokenizer().tokenize(true_test[batch_i * self.batch_size + k])])

            bleu_scores = calculate_bleu_scores(references_test, hypotheses_test)

        print('BLEU 1 to 4 : {}'.format(' | '.join(map(str, bleu_scores))))

        return pred_logits

    def sample(self):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope('sample_normal'):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(self.z_log_sigma), name='epsilon')

            if self.distribution == 'dirichlet':
                return self.z_mean + tf.scalar_mul(self.z_temperature, epsilon * self.z_log_sigma)
            else:
                # N(mu, I * sigma**2)
                return self.z_mean + tf.scalar_mul(self.z_temperature, epsilon * tf.exp(self.z_log_sigma))

    def calculate_kl_loss(self):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            if self.distribution == 'dirichlet':
                a0 = tf.reduce_sum(self.z_log_alpha, axis=-1, keepdims=True)
                return (
                        tf.lgamma(a0) -
                        tf.reduce_sum(tf.lgamma(self.z_log_alpha), 1) +
                        tf.reduce_sum(self.z_log_alpha * tf.digamma(self.z_log_alpha), 1) -
                        tf.reduce_sum(self.z_log_alpha * tf.digamma(a0), 1) -
                        tf.reduce_mean(tf.digamma(self.z_log_alpha) - tf.digamma(a0), 1)
                )
            else:
                # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
                return -0.5 * tf.reduce_sum(
                    1.0 + 2 * self.z_log_sigma - self.z_mean ** 2 - tf.exp(2 * self.z_log_sigma), 1)

    def show_output_sentences(self, preds, y_test, input_test, true_test, to=None):
        symbol = []
        if self.config['experiment'] == 'qgen':
            symbol.append('?')

        df_dict = {'input': [], 'actual': [], 'generated': []}

        for k, (pred, actual) in enumerate(zip(preds, y_test)):
            _input = input_test[k].strip()
            _actual = true_test[k].strip()
            _generated = " ".join([self.decoder_idx_word[i] for i in pred if i not in [self.pad, self.eos]] + symbol)

            df_dict['input'].append(_input)
            df_dict['actual'].append(_actual)
            df_dict['generated'].append(_generated)

            print('Input:      {}'.format(self.wrapper.fill(_input)))
            print('Actual:     {}'.format(self.wrapper.fill(_actual)))
            print('Generated: {}\n'.format(self.wrapper.fill(_generated)))

        if isinstance(to, str):
            df_dict = pd.DataFrame.from_dict(df_dict)
            df_dict.to_csv(to)

    def get_diversity_metrics(self, checkpoint, x_test, y_test, num_samples=10, num_iterations=3):

        x_test_repeated = np.repeat(x_test, num_samples, axis=0)
        y_test_repeated = np.repeat(y_test, num_samples, axis=0)

        entropy_list = []
        uni_diversity = []
        bi_diversity = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            for _ in tqdm(range(num_iterations)):
                total_ent = 0
                uni = 0
                bi = 0
                answer_logits = []
                pred_sentences = []

                for batch_i, (input_batch, output_batch, source_sent_lengths, tar_sent_lengths) in enumerate(
                        get_batches(x_test_repeated, y_test_repeated, self.batch_size)):
                    result = sess.run(self.inference_logits, feed_dict={
                        self.input_data: input_batch,
                        self.source_sentence_length: source_sent_lengths,
                        self.keep_prob: 1.0,
                        self.word_dropout_keep_prob: 1.0,
                        self.z_temperature: self.z_temp,
                        self.attention_temperature: self.attention_temp})
                    answer_logits.extend(result)

                for idx, (actual, pred) in enumerate(zip(y_test_repeated, answer_logits)):
                    pred_sentences.append(" ".join([self.decoder_idx_word[i] for i in pred if i != self.pad][:-1]))

                    if (idx + 1) % num_samples == 0:
                        word_list = [WhitespaceTokenizer().tokenize(p) for p in pred_sentences]
                        corpus = [item for sublist in word_list for item in sublist]
                        total_ent += calculate_entropy(corpus)
                        diversity_result = calculate_ngram_diversity(corpus)
                        uni += diversity_result[0]
                        bi += diversity_result[1]

                        pred_sentences = []

                entropy_list.append(total_ent / len(x_test))
                uni_diversity.append(uni / len(x_test))
                bi_diversity.append(bi / len(x_test))

        print()
        print('Entropy = {:>.3f} | Distinct-1 = {:>.3f} | Distinct-2 = {:>.3f}'.format(
            np.mean(entropy_list),
            np.mean(uni_diversity),
            np.mean(bi_diversity)))
