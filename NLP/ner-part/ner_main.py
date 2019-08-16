#this main file only include the predict part, if you want to train this model again,please see BERT-NER.py
# -*- coding:utf-8 -*-

from BERT_NER import NerProcessor,model_fn_builder,file_based_input_fn_builder
from utils import filed_based_convert_examples_to_features
import tensorflow as tf
from bert import modeling
from bert import optimization
from bert import tokenization
import os
import pickle
import collections


mainroot = os.path.dirname(os.path.realpath(__file__))

def NER(max_seq_length = 128,
        bert_config_file = mainroot + '/checkpoint/bert_config.json',
        task_name = 'NER',
        vocab_file = mainroot + '/vocab.txt',
        do_lower_case = True,
        output_dir = mainroot + '/bert/output/result_dir/',
        test_file_path = '',
        test_word = '',
        init_checkpoint = mainroot + '/bert/output/result_dir/model.ckpt-1524',
        # use_file_or_string = 0,
        input_string = ''):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=1,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=init_checkpoint,
        learning_rate=5e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=8)

    token_path = os.path.join(output_dir, "token_test.txt")
    with open(mainroot + '/output/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    if os.path.exists(token_path):
        os.remove(token_path)

    if input_string != '':
        predict_examples = processor.get_test_examples(input_string=input_string,
                                                       input_str=True)
    elif test_file_path != '':
       predict_examples = processor.get_test_examples(test_file_path, single_file=True)
    else:
        print("input string or test file path must have one not be none!")
        return
    print(predict_examples)
    predict_file = os.path.join(output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             max_seq_length, tokenizer,
                                             predict_file,output_dir, mode="test")

    # tf.logging.info("***** Running prediction*****")
    # tf.logging.info("  Num examples = %d", len(predict_examples))
    # tf.logging.info("  Batch size = %d", 8)
    # if FLAGS.use_tpu:
    #     # Warning: According to tpu_estimator.py Prediction on TPU is an
    #     # experimental feature and hence not supported here
    #     raise ValueError("Prediction in TPU not supported")
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    tf.reset_default_graph()
    sess = tf.Session()
    sess.close()
    sess._closed
    sess._opened
    # print(result)
    Org =[]
    Time = [] 
    Loc = []
    output_predict_file = os.path.join(output_dir, "label_test.txt")
    strings = input_string.split('\n')
    num = 0
    word =0
    flag = ''
    name = ''
    with open(output_predict_file, 'w') as writer:
        for prediction in result:
            output_line = "\n".join(id2label[id] for id in prediction if id != 0) + "\n"
            print(output_line.split('\n')[1:-1])
            output_line1 = output_line.split('\n')[1:-1]
            print(strings[num])
            for i in range(len(output_line1)):
                # print(output_line1[i])
                if output_line1[i] == 'B-Organization' or output_line1[i] == 'B-Time' \
                        or output_line1[i] == 'B-Location':
                    name = output_line1[i]
                    flag = strings[num][i]
                    # print(strings[num][i])
                    # print(name)
                elif output_line1[i] == 'I-Organization' or output_line1[i] == 'I-Time' or output_line1[i] == 'I-Location':
                    if name != '':
                        flag = flag + strings[num][i]
                elif output_line1[i] == 'O' or output_line1[i] == '[SEP]':
                    print(name)
                    if name == 'B-Organization' : Org.append(flag)
                    elif name == 'B-Time' : Time.append(flag)
                    elif name == 'B-Location': Loc.append(flag)
                    flag = ''
                    name = ''
            num = num + 1

            writer.write(output_line)
        print(Org,Time,Loc)
        if len(Org) >= 1: org = max(Org)
        else:org = ''
        if len(Time) >= 1: tim = max(Time)
        else:tim = ''
        if len(Loc) >= 1: loc = max(Loc)
        else:loc = ''

        return org,tim,loc

if __name__ == '__main__':
    string = open(os.path.abspath('..')+'/CarReport_6.txt','r',encoding='utf-8').read()
    # print(string)
    org,tim,loc = NER(input_string=string)
    print(org,tim,loc)