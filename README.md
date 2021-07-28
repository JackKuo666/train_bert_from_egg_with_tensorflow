# 这里是从头训练bert模型的tensorflow版本
# pip
注意：这里需要使用TensorFlow==2.5.0
需要以下依赖：
```buildoutcfg
# tensorflow-gpu  >= 2.5.0  # GPU version of TensorFlow.
tensorflow >= 2.5.0   # CPU Version of TensorFlow.
gin-config==0.1.1
tensorflow_hub
tensorflow_addons
```

# step 0: create vocab.txt
```buildoutcfg
python 0.make_vocab.py
```
这里会在`bert-min`文件夹中生成`vocab.txt`
# step 1.构造10fold数据
将数据按类别分类，然后再平均分到10份中，这样没份中的各个类别就比较均衡
```buildoutcfg
python 1.构造10fold数据.py
```
这里会在`train_split_data`中产生经过10fold的data
# step 2: create pretraining data
注意，这里windows下需要一个一个修改文件运行，具体可以看2.create_pretraining_data.py：main下的代码
linux 可以使用shell 脚本直接运行
```buildoutcfg
python 2.create_pretraining_data.py 
```
这里会在`records/`中产生10个tfrecords数据

# step 3: run pretraining
首先需要在`bert-mini`中放置`config.json`来配置要训练的bert的参数
```buildoutcfg
python 3.run_pretraining.py --input_files=./records/*.tfrecord --model_export_path=./checkpoint --bert_config_file=./bert-mini/config.json --train_batch_size=128 --max_seq_length=256 --max_predictions_per_seq=32 --learning_rate=1e-4
```
# step 4: convert checkpoint

```buildoutcfg
python 4.convert_checkpoint.py --tf_checkpoint checkpoint/bert_model.ckpt-8 --bert_config_file bert-mini/config.json --pytorch_dump_path bert-mini/pytorch_model.bin
```