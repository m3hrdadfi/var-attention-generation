from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import unicodedata
from .contractions import fix


def tokenize_sequence(sentences, filters, max_num_words, max_vocab_size):
    """
    Tokenize a given input sequence of words.

    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary

    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence

    """

    sentences = [' '.join(WhitespaceTokenizer().tokenize(s)[:max_num_words]) for s in sentences]

    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)

    word_index = dict()
    word_index['PAD'] = 0
    word_index['UNK'] = 1
    word_index['GO'] = 2
    word_index['EOS'] = 3

    for i, word in enumerate(dict(tokenizer.word_index).keys()):
        word_index[word] = i + 4

    tokenizer.word_index = word_index
    x = tokenizer.texts_to_sequences(list(sentences))

    for i, seq in enumerate(x):
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['UNK'] for t in seq]
        seq.append(word_index['EOS'])
        x[i] = seq

    x = pad_sequences(x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['PAD'])

    word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}

    return x, word_index, sentences


def create_embedding_matrix(word_index, embedding_dim, w2v_path):
    """
    Create the initial embedding matrix for TF Graph.
    Args:
        word_index: dictionary storing the word-to-index correspondence
        embedding_dim: word2vec dimension
        w2v_path: file path to the w2v pickle file
    Returns:
        embeddings_matrix : numpy 2d-array with word vectors
    """
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index), embedding_dim))

    if isinstance(w2v_path, str):
        w2v_model = gensim.models.Word2Vec.load(w2v_path)
        for word, i in word_index.items():
            try:
                embeddings_vector = w2v_model[word]
                embeddings_matrix[i] = embeddings_vector
            except KeyError:
                pass
    elif isinstance(w2v_path, dict):
        for word, i in word_index.items():
            try:
                embedding_vector = w2v_path.get(word)
                if embedding_vector is not None:
                    embeddings_matrix[i] = embedding_vector
            except KeyError:
                pass
    else:
        raise ValueError('`w2v_path` must be string or dict.')

    return embeddings_matrix


def excerpt(text):
    text = sent_tokenize(text)
    if len(text) > 0:
        return text[0]
    return None


def cleanup_text(text):
    if not isinstance(text, str):
        print('text', text)

    cleaner = re.compile('<.*?>')
    text = unicodedata.normalize('NFKD', text)
    text = fix(text)
    text = re.sub(cleaner, '', text)
    return text


def noise_reduction(row):
    content_tokens = WhitespaceTokenizer().tokenize(str(row['content']))
    title_tokens = WhitespaceTokenizer().tokenize(str(row['title']))

    conditions_1 = len(content_tokens) < len(title_tokens)
    conditions_2 = len(content_tokens) < 5
    conditions_3 = len(title_tokens) < 3
    conditions = conditions_1 or conditions_2 or conditions_3
    if conditions:
        return None

    return 1


def create_news_data(sources, num_samples=None, preprocessing=True):
    dfs = []

    for source in sources:
        df = pd.read_csv(source)
        df = df[['content', 'title', 'publication']]
        df = df.drop_duplicates(subset=['title'], keep=False)

        dfs.append(df)

    data = pd.concat(dfs)
    data = data.dropna()
    data = data.reset_index(drop=True)

    data = data.drop_duplicates(subset=['title'], keep=False)
    data = data.sample(frac=1).reset_index(drop=True)

    del dfs

    if num_samples:
        data = data[:num_samples]

    data['content'] = data['content'].apply(lambda x: excerpt(x))
    data = data.dropna()
    data = data.reset_index(drop=True)

    if preprocessing:
        data['content'] = data['content'].apply(lambda x: cleanup_text(x))
        data['title'] = data['title'].apply(lambda x: cleanup_text(x))
        data['condition'] = data.apply(noise_reduction, axis=1)

        data = data.dropna()
        data = data.reset_index(drop=True)

    data['content'] = data['content'].astype(str)
    data['title'] = data['title'].astype(str)
    data = data[['content', 'title', 'publication']]

    return data


def create_data_split(x, y, valid_size=.2, test_size=.5, verbose=False, random_state=101):
    x_sen_train = y_sen_train = None
    x_sen_valid = y_sen_valid = None
    x_sen_test = y_sen_test = None

    if isinstance(x, list) and isinstance(y, list):
        if not len(x[1]) == len(y[1]):
            raise ValueError('X, Y data must be equal.')

        idx = list(range(len(x[1])))

        train_idx, _idx = train_test_split(idx, test_size=valid_size, random_state=random_state)
        valid_idx, test_idx = train_test_split(_idx, test_size=test_size, random_state=random_state)

        x_train = x[0][train_idx]
        y_train = y[0][train_idx]
        x_sen_train = np.array(x[1])[train_idx]
        y_sen_train = np.array(y[1])[train_idx]

        x_valid = x[0][valid_idx]
        y_valid = y[0][valid_idx]
        x_sen_valid = np.array(x[1])[valid_idx]
        y_sen_valid = np.array(y[1])[valid_idx]

        x_test = x[0][test_idx]
        y_test = y[0][test_idx]
        x_sen_test = np.array(x[1])[test_idx]
        y_sen_test = np.array(y[1])[test_idx]


    elif isinstance(x, list) or isinstance(y, list):
        raise ValueError('X, Y data must be both in nd.array or list type.')

    else:
        x_train, _x, y_train, _y = train_test_split(x, y, test_size=valid_size, random_state=random_state)
        x_valid, x_test, y_valid, y_test = train_test_split(_x, _y, test_size=test_size, random_state=random_state)

    if verbose:
        print('[INFO] Training ...')
        print('X Shape:', x_train.shape)
        print('Y Shape:', y_train.shape)
        print()

        print('[INFO] Validating ...')
        print('X Shape:', x_valid.shape)
        print('Y Shape:', y_valid.shape)
        print()

        print('[INFO] Testing ...')
        print('X Shape:', x_test.shape)
        print('Y Shape:', y_test.shape)
        print()

    train = (x_train, y_train, x_sen_train, y_sen_train)
    valid = (x_valid, y_valid, x_sen_valid, y_sen_valid)
    test = (x_test, y_test, x_sen_test, y_sen_test)
    return train, valid, test


def get_batches(x, y, batch_size):
    """
    Generate inputs and targets in a batch-wise fashion for feed-dict

    Args:
        x: entire source sequence array
        y: entire output sequence array
        batch_size: batch size

    Returns:
        x_batch, y_batch, source_sentence_length, target_sentence_length

    """

    for batch_i in range(0, len(x) // batch_size):
        start_i = batch_i * batch_size
        x_batch = x[start_i:start_i + batch_size]
        y_batch = y[start_i:start_i + batch_size]

        source_sentence_length = [np.count_nonzero(seq) for seq in x_batch]
        target_sentence_length = [np.count_nonzero(seq) for seq in y_batch]

        yield x_batch, y_batch, source_sentence_length, target_sentence_length
