2 epock 1 trial

adrian@adrian-Genie-VIG880S:~/projects/eeg_visual_classification$ python3 eeg_signal_classification_linux.py 
config: {'input_size': 128, 'lstm_size': 128, 'lstm_layers': <ray.tune.sample.Function object at 0x7f43a50fcf98>, 'output_size': 128, 'lr': <ray.tune.sample.Float object at 0x7f43a3915940>}
Namespace(data_workers=4, epochs=200, learning_rate_decay_by=0.5, learning_rate_decay_every=10, model_type='lstm', no_cuda=False, pretrained_net='', saveCheck=100, split_num=0, subject=0, time_high=480, time_low=320)
2021-04-22 11:10:18,334	INFO services.py:1092 -- View the Ray dashboard at http://127.0.0.1:8265
== Status ==
Memory usage on this node: 4.9/15.6 GiB
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 2.000: None | Iter 1.000: None
Resources requested: 12/12 CPUs, 2/2 GPUs, 0.0/7.03 GiB heap, 0.0/2.39 GiB objects (0/1.0 accelerator_type:GTX)
Result logdir: /home/adrian/ray_results/inner_2021-04-22_11-10-19
Number of trials: 1/1 (1 RUNNING)
+-------------------+----------+-------+------------+---------------+
| Trial name        | status   | loc   |         lr |   lstm_layers |
|-------------------+----------+-------+------------+---------------|
| inner_f0f9b_00000 | RUNNING  |       | 0.00153055 |             4 |
+-------------------+----------+-------+------------+---------------+


Result for inner_f0f9b_00000:
  accuracy: 0.02217741935483871
  date: 2021-04-22_11-10-49
  done: false
  experiment_id: 9bd8215f42a74aeebbbd935aac4e46a9
  experiment_tag: 0_lr=0.0015305,lstm_layers=4
  hostname: adrian-Genie-VIG880S
  iterations_since_restore: 1
  loss: 3.6910482375852522
  node_ip: 192.168.0.16
  pid: 29201
  should_checkpoint: true
  time_since_restore: 29.113648176193237
  time_this_iter_s: 29.113648176193237
  time_total_s: 29.113648176193237
  timestamp: 1619086249
  timesteps_since_restore: 0
  training_iteration: 1
  trial_id: f0f9b_00000
  
== Status ==
Memory usage on this node: 9.4/15.6 GiB
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 2.000: None | Iter 1.000: -3.6910482375852522
Resources requested: 12/12 CPUs, 2/2 GPUs, 0.0/7.03 GiB heap, 0.0/2.39 GiB objects (0/1.0 accelerator_type:GTX)
Current best trial: f0f9b_00000 with loss=3.6910482375852522 and parameters={'input_size': 128, 'lstm_size': 128, 'lstm_layers': 4, 'output_size': 128, 'lr': 0.00153054987009036}
Result logdir: /home/adrian/ray_results/inner_2021-04-22_11-10-19
Number of trials: 1/1 (1 RUNNING)
+-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------+
| Trial name        | status   | loc                |         lr |   lstm_layers |   iter |   total time (s) |    loss |   accuracy |
|-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------|
| inner_f0f9b_00000 | RUNNING  | 192.168.0.16:29201 | 0.00153055 |             4 |      1 |          29.1136 | 3.69105 |  0.0221774 |
+-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------+


Result for inner_f0f9b_00000:
  accuracy: 0.017641129032258066
  date: 2021-04-22_11-11-13
  done: true
  experiment_id: 9bd8215f42a74aeebbbd935aac4e46a9
  experiment_tag: 0_lr=0.0015305,lstm_layers=4
  hostname: adrian-Genie-VIG880S
  iterations_since_restore: 2
  loss: 3.6940780205111348
  node_ip: 192.168.0.16
  pid: 29201
  should_checkpoint: true
  time_since_restore: 53.09364914894104
  time_this_iter_s: 23.980000972747803
  time_total_s: 53.09364914894104
  timestamp: 1619086273
  timesteps_since_restore: 0
  training_iteration: 2
  trial_id: f0f9b_00000
  
== Status ==
Memory usage on this node: 9.4/15.6 GiB
Using AsyncHyperBand: num_stopped=1
Bracket: Iter 2.000: None | Iter 1.000: -3.6910482375852522
Resources requested: 12/12 CPUs, 2/2 GPUs, 0.0/7.03 GiB heap, 0.0/2.39 GiB objects (0/1.0 accelerator_type:GTX)
Current best trial: f0f9b_00000 with loss=3.6940780205111348 and parameters={'input_size': 128, 'lstm_size': 128, 'lstm_layers': 4, 'output_size': 128, 'lr': 0.00153054987009036}
Result logdir: /home/adrian/ray_results/inner_2021-04-22_11-10-19
Number of trials: 1/1 (1 RUNNING)
+-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------+
| Trial name        | status   | loc                |         lr |   lstm_layers |   iter |   total time (s) |    loss |   accuracy |
|-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------|
| inner_f0f9b_00000 | RUNNING  | 192.168.0.16:29201 | 0.00153055 |             4 |      2 |          53.0936 | 3.69408 |  0.0176411 |
+-------------------+----------+--------------------+------------+---------------+--------+------------------+---------+------------+


== Status ==
Memory usage on this node: 9.4/15.6 GiB
Using AsyncHyperBand: num_stopped=1
Bracket: Iter 2.000: None | Iter 1.000: -3.6910482375852522
Resources requested: 0/12 CPUs, 0/2 GPUs, 0.0/7.03 GiB heap, 0.0/2.39 GiB objects (0/1.0 accelerator_type:GTX)
Current best trial: f0f9b_00000 with loss=3.6940780205111348 and parameters={'input_size': 128, 'lstm_size': 128, 'lstm_layers': 4, 'output_size': 128, 'lr': 0.00153054987009036}
Result logdir: /home/adrian/ray_results/inner_2021-04-22_11-10-19
Number of trials: 1/1 (1 TERMINATED)
+-------------------+------------+-------+------------+---------------+--------+------------------+---------+------------+
| Trial name        | status     | loc   |         lr |   lstm_layers |   iter |   total time (s) |    loss |   accuracy |
|-------------------+------------+-------+------------+---------------+--------+------------------+---------+------------|
| inner_f0f9b_00000 | TERMINATED |       | 0.00153055 |             4 |      2 |          53.0936 | 3.69408 |  0.0176411 |
+-------------------+------------+-------+------------+---------------+--------+------------------+---------+------------+


2021-04-22 11:11:13,069	INFO tune.py:439 -- Total run time: 55.28 seconds (54.01 seconds for the tuning loop).
Best trial config: {'input_size': 128, 'lstm_size': 128, 'lstm_layers': 4, 'output_size': 128, 'lr': 0.00153054987009036}
Best trial final validation loss: 3.6940780205111348
Best trial final validation accuracy: 0.017641129032258066
Best trial test set accuracy: 0.012096774193548387