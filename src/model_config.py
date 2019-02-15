config = dict(
    distribution='dirichlet',  # gauss or dirichlet
    model='VarSeq2SeqVarAttnModel',
    experiment='?',
    num_samples=10000,
    preprocessing=True,
    lstm_hidden_units=300,  # Number of hidden units for the LSTM
    embedding_size=300,  # Word embedding dimension
    num_layers=1,  # Number of LSTM layers

    encoder_vocab=40000,  # Vocabulary size on the encoder side # 30000 for dialogue
    decoder_vocab=40000,  # Vocabulary size on the decoder side # 30000 for dialogue
    encoder_num_tokens=40,  # Number of words/tokens in the input sequence # 20 for dialogue
    decoder_num_tokens=30,  # Number of words/tokens in the generated sequence

    dropout_keep_prob=0.8,  # Dropout keep probability
    initial_learning_rate=0.001,  # Initial learning rate
    learning_rate_decay=0.75,  # Learning rate decay
    min_learning_rate=0.00001,  # Minimum learning rate

    latent_dim=100,  # Dimension of z-latent space
    word_dropout_keep_probability=0.8,  # 1.0 - Word dropout rate for the decoder
    z_temp=1.1,  # Sampling temperature to be multiplied with the standard deviation
    attention_temp=1.1,  # Sampling temperature for the attention/context vector
    use_hmean=True,  # Whether to use N(h_mean, I) or N(0, I) as the prior
    gamma_val=0.1,  # Coefficient of the attention KL loss

    batch_size=100,  # Batch size # 128 for dialogue
    n_epochs=10,  # Number of epochs

    logs_dir='./outputs/ved-var-attn/summary/',  # Path to save summary information for Tensorboard
    log_str_dir='./outputs/ved-var-attn/logs/',  # Path to save summary information for Tensorboard
    model_checkpoint_dir='./outputs/ved-var-attn/checkpoints/ckps-',  # Path to save model checkpoints
    bleu_path='./outputs/outputs/ved-var-attn/bleu/det',  # Path to save model checkpoints
    w2v_dir='./outputs/data/',  # Word2Vec model directory
    data_dir='./data/',  # Directory to store data csv files

    load_checkpoint=7,
    # Specify the trained model epoch/checkpoint number to be loaded for evaluation on test set,
    # 0 means last saved checkpoint
    wrapper=None,
)
