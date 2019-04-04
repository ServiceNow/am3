# ADAPTIVE CROSS-MODAL FEW-SHOT LEARNING (AW3)

Code for paper Adaptive Cross-Modal Few-shot Learning. [[Arxiv]](http://www.dropwizard.io/1.0.2/docs/)

## Dependencies

* cv2
* numpy
* python 3.5+
* tensorflow 1.3+
* tqdm
* scipy

## Datasets

First, designate a folder to be your data root:
```
export DATA_ROOT={DATA_ROOT}
Then, set up the datasets following the instructions in the subsections.
```
###miniImageNet

[[Google Drive]](https://drive.google.com/file/d/1g4wOa0FpWalffXJMN2IZw0K2TM2uxzbk/view?usp=sharing)(1.05G)
```
# Download and place "mini-imagenet.zip" in "$DATA_ROOT/mini-imagenet".
mkdir -p $DATA_ROOT/mini-imagenet
cd $DATA_ROOT/mini-imagenet
mv ~/Downloads/mini-imagenet.zip .
unzip mini-imagenet.zip
rm -f mini-imagenet.zip
```
###tieredImageNet
[[Google Drive]](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)(14.33G)
```
# Download and place "tiered-imagenet.zip" in "$DATA_ROOT/tiered-imagenet".
mkdir -p $DATA_ROOT/tiered-imagenet
cd $DATA_ROOT/tiered-imagenet
mv ~/Downloads/tiered-imagenet.tar.gz .
tar -xvf tiered-imagenet.tar.gz
rm -f tiered-imagenet.tar.gz
```
## AM3-ProtoNet
### 1-shot experiments
For mini-ImageNet:
```
python AM3_protonet++.py --data_dir $DATA_ROOT/mini-imagenet/ 
--num_tasks_per_batch 5 --num_shots_train 1 --num_shots_test 1 --train_batch_size 24 
--mlp_dropout 0.7 --att_input word --task_encoder self_att_mlp 
--mlp_type non-linear --mlp_weight_decay 0.001
--log_dir $EXP_DIR
```

For tiered-ImageNet:
```
python AM3_protonet++.py --data_dir $DATA_ROOT/tiered-imagenet/ 
--num_tasks_per_batch 5 --num_shots_train 1 --num_shots_test 1 --train_batch_size 24
--num_steps_decay_pwc 10000 --number_of_steps 80000  
--mlp_dropout 0.7 --att_input word --task_encoder self_att_mlp 
--mlp_type non-linear --mlp_weight_decay 0.001
--log_dir $EXP_DIR

```

### 5-shot experiments
For mini-ImageNet:
```
python AM3_protonet++.py --data_dir $DATA_ROOT/mini-imagenet/  
--mlp_dropout 0.7 --att_input word --task_encoder self_att_mlp 
--mlp_type non-linear --mlp_weight_decay 0.001
--log_dir $EXP_DIR
```

For tiered-ImageNet:
```
python AM3_protonet++.py --data_dir $DATA_ROOT/tiered-imagenet/ 
--num_steps_decay_pwc 10000 --number_of_steps 80000 
--mlp_dropout 0.7 --att_input word --task_encoder self_att_mlp 
--mlp_type non-linear --mlp_weight_decay 0.001
--log_dir $EXP_DIR

```

##AM3-TADAM
Note that you may need to tune "--metric_multiplier_init" which is a TADAM hyper-parameter, via cross-validation to achieve sota results. The range of "--metric_multiplier_init" is usually (5, 10). 
### 1-shot experiments
For mini-ImageNet:
```
python AM3_TADAM.py --data_dir $DATA_ROOT/mini-imagenet/ 
--num_tasks_per_batch 5 --num_shots_train 1 --num_shots_test 1 --train_batch_size 24 --metric_multiplier_init 5
--feat_extract_pretrain multitask --encoder_classifier_link cbn --num_cases_test 100000 
--activation_mlp relu --att_dropout 0.7 --att_type non-linear --att_weight_decay 0.001 
--mlp_dropout 0.7 --mlp_type non-linear --mlp_weight_decay 0.001 --att_input word --task_encoder self_att_mlp 
--log_dir $EXP_DIR
```
For tiered-ImageNet:
```
python AM3_TADAM.py --data_dir $DATA_ROOT/tiered-imagenet/ 
--num_tasks_per_batch 5 --num_shots_train 1 --num_shots_test 1 --train_batch_size 24 --metric_multiplier_init 5
--feat_extract_pretrain multitask --encoder_classifier_link cbn --num_steps_decay_pwc 10000 
--number_of_steps 80000 --num_cases_test 100000 --num_classes_pretrain 351 
--att_dropout 0.9  --mlp_dropout 0.9 
--log_dir "$EXP_DIR

```

### 5-shot experiments
For mini-ImageNet:
```
python AM3_TADAM.py --data_dir $DATA_ROOT/mini-imagenet/ 
--metric_multiplier_init 7
--feat_extract_pretrain multitask --encoder_classifier_link cbn --num_cases_test 100000 
--activation_mlp relu --att_dropout 0.7 --att_type non-linear --att_weight_decay 0.001 
--mlp_dropout 0.7 --mlp_type non-linear --mlp_weight_decay 0.001 --att_input word --task_encoder self_att_mlp 
--log_dir $EXP_DIR
```
For tiered-ImageNet:
```
python AM3_TADAM.py --data_dir $DATA_ROOT/tiered-imagenet/ 
--metric_multiplier_init 7
--feat_extract_pretrain multitask --encoder_classifier_link cbn --num_steps_decay_pwc 10000 
--number_of_steps 80000 --num_cases_test 100000 --num_classes_pretrain 351 
--att_dropout 0.9  --mlp_dropout 0.9 
--log_dir "$EXP_DIR

```

## Citation

If you use our code, please consider cite the following:

* Chen Xing,

