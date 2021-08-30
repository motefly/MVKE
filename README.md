# MVKE
Implementation for the paper "Mixture of Virtual-Kernel Experts for Multi-Objective UserProfile Modeling", which has been submitted to WSDM'2022.

This repo releases part of core codes for the implementation of MVKE and baselines. The loss definitions are set in "model_graph_type.py" and the detailed model architecture is defined in "mvke_ops.py". More codes is highly related to a complex industrial learning architecture, thus it is temporarily inconvenient to open source.

The following table lists the loss entry name of methods:

| Function | Description|
|--|--|
| user_MVKE_and_tag_HS | The user tower is equipped with MVKE and tag tower adopts Hard-sharing structure |
| user_MMoE_and_tag_HS | The user tower is equipped with MMoE/CGC and tag tower adopts Hard-sharing structure |
| user_HS_and_tag_HS | Both user tower and tag tower adopts Hard-sharing structure |
| user_and_tag_single_pctr/pcvr | Only test the performance on single task |


Thanks for your visiting, and if you have any questions, please new an issue.