# Multi-Objective Evolutionary Optimization of Imbalanced Fast Feedforward Networks

## Experiments

### Training an Imbalanced FFF

Run the following command to train an imbalanced Fast Feedforward (FFF) network with multiple seeds:

```bash
python src/main.py --multirun exp=train_anyddt seed=1,2,3,4
```

To modify the architecture, edit `conf/train_anyddt.yaml`:

```yaml
tree: 
  _target_: utils.Tree
  arch: [0, 4, 0, -1, -1, 8, 1]  # Unbalanced architecture, d=2; leaf if >= 1
```

For an IFFF depth of 2, the above architecture is divided by depth as:

```
(0) - (4, 0) - (-1, -1, 8, 1)
```

- `0`: internal node
- `-1`: none
- `x > 0`: leaf node with width `x`

Visualized, the tree structure is:

```
    o
   / \
 [4]  o
     / \
   [8] [1]
```

### Evolving an Imbalanced FFF

To evolve using Grammatical Evolution (GE):

```bash
python src/main.py --multirun exp=nas_ddt_ge loader=mnist,sc,motionsense
```

To evolve using the default encoding:

```bash
python src/main.py --multirun exp=nas_ddt loader=mnist,sc,motionsense
```

## Datasets:

In addition to very well known sets as MNIST, SpeechCommandsv2, we also provide a HAR dataset:

### [Motion Sense Dataset](https://github.com/mmalekzadeh/motion-sense)

Thus, we have time-series with 12 features:

    attitude.roll
    attitude.pitch
    attitude.yaw
    gravity.x
    gravity.y
    gravity.z
    rotationRate.x
    rotationRate.y
    rotationRate.z
    userAcceleration.x
    userAcceleration.y
    userAcceleration.z

    
Data from 24 subjects performing 6 activities:            
dws: downstairs
ups: upstairs
sit: sitting
std: standing
wlk: walking
jog: jogging

## Citation
If you find this work useful, please consider citing:
```bibtex
@InProceedings{10.1007/978-3-032-23604-3_23,
author="Kilic, Renan Beran
and Yildirim, Kasim Sinan
and Iacca, Giovanni",
editor="Garc{\'i}a-S{\'a}nchez, Pablo
and D{\'i}az {\'A}lvarez, Josefa
and Murphy, Aidan",
title="Multi-objective Evolutionary Optimization of Imbalanced Fast Feedforward Networks",
booktitle="Applications of Evolutionary Computation",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="367--383",
isbn="978-3-032-23604-3"
}
```
