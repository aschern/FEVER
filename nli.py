import sys

#!test -d bert_repo || git clone https://github.com/google-research/bert bert_repo
if not 'bert_repo' in sys.path:
    sys.path += ['bert_repo']
    
import datetime
import json
import logging
import modeling
import optimization
import os
import pprint
import random
import run_classifier
import string
import sys
import tensorflow as tf
import tokenization


logging.getLogger().setLevel(logging.INFO)

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
logging.info('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

OUTPUT_DIR = 'results/'


TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 1.3
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
NUM_TPU_CORES = 8
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')


class FeverProcessor(run_classifier.DataProcessor):
    """Processor for the FEVER data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eval.tsv")), #dev
            "dev_matched")

    def get_test_examples(self, data_dir, path_pred):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, path_pred)), "test")

    def get_labels(self):
        """See base class."""
        return ["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or line == []:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = "NOT ENOUGH INFO"
            else:
                label = tokenization.convert_to_unicode(line[2])
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
def bert_model(task_type, file_type=None):
    '''
    Args:
    task_type - "train", "eval", "test"
    file_type - 'test', 'eval' or 'demo' for prediction
    '''
    assert task_type in ['test', 'train', 'eval'], "the 'task_type' parameter must take one of four values: 'test', 'train' or 'eval'"
    assert task_type != 'test' or file_type in ['test', 'demo', 'eval'], 'you must select a file type ("demo", "eval" or "test") in the case of testing'
    
    logging.info('preprocessing')
    
    processor = FeverProcessor()
    label_list = processor.get_labels()
    
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
    
    tpu_cluster_resolver = None
    run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    log_step_count_steps=100,
    keep_checkpoint_max=1,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
    
    #tf.logging.set_verbosity(tf.logging.INFO)
    
    if task_type == 'train':
        train_examples = processor.get_train_examples(OUTPUT_DIR)
    else:
        train_examples = ['pad']
    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    if task_type == 'train':
        INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    else:
        for file in os.listdir('results'):
            if file.endswith(".meta"):
                INIT_CHECKPOINT = os.path.join('results', file[:-5])
                break
            os.path.join('results', check)
    
    model_fn = run_classifier.model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE)
    
    logging.info('done')
    
    if task_type == 'train':
        train_features = run_classifier.convert_examples_to_features(
        train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        print('***** Started training at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(train_examples)))
        print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)  #False
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))
        
    if task_type == 'eval':
        eval_examples = processor.get_dev_examples(OUTPUT_DIR)
        eval_features = run_classifier.convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
        print('  Num examples = {}'.format(len(eval_examples)))
        print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
        # Eval will be slightly WRONG on the TPU because it will truncate
        # the last batch.
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_input_fn = run_classifier.input_fn_builder(
            features=eval_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))

    if task_type == 'test':
        PREDICT_BATCH_SIZE = 5
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            predict_batch_size=PREDICT_BATCH_SIZE
        )
        
        predict_examples = processor.get_test_examples(OUTPUT_DIR, 'pred_{}.tsv'.format(file_type))
            
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(OUTPUT_DIR, "predict.tf_record")
        run_classifier.file_based_convert_examples_to_features(predict_examples, label_list,
                                                MAX_SEQ_LENGTH, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", PREDICT_BATCH_SIZE)

        predict_drop_remainder = True
        predict_input_fn = run_classifier.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(OUTPUT_DIR, "results_{}.csv".format(file_type))

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = ",".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
