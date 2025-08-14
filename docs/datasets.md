# Datasets
This page provides an overview of the datasets available in the Dronalize toolbox for trajectory prediction. 
It contains details on each dataset, including links to their respective publications, abstracts, BibTeX entries, preprocessing instructions, and a brief overview of the dataset characteristics.

---

### *[Argoverse](https://arxiv.org/abs/1911.02620)*: 3D Tracking and Forecasting with Rich Maps

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    We present Argoverse – two datasets designed to support autonomous vehicle machine learning tasks such as 3D tracking and motion forecasting. Argoverse was collected by a fleet of autonomous vehicles in Pittsburgh and Miami. The Argoverse 3D Tracking dataset includes 360 degree images from 7 cameras with overlapping fields of view, 3D point clouds from long range LiDAR, 6-DOF pose, and 3D track annotations. Notably, it is the only modern AV dataset that provides forward-facing stereo imagery. The Argoverse Motion Forecasting dataset includes more than 300,000 5-second tracked scenarios with a particular vehicle identified for trajectory forecasting. Argoverse is the first autonomous vehicle dataset to include "HD maps" with 290 km of mapped lanes with geometric and semantic metadata. All data is released under a Creative Commons license at http://www.argoverse.org/. In our baseline experiments, we illustrate how detailed map information such as lane direction, driveable area, and ground height improves the accuracy of 3D object tracking and motion forecasting. Our tracking and forecasting experiments represent only an initial exploration of the use of rich maps in robotic perception. We hope that Argoverse will enable the research community to explore these problems in greater depth.
  </p>
</details>

<details>
  <summary>Bibtex</summary>
  
```bibtex
  @inproceedings{chang2019argoverse,
    title={Argoverse: {3D} tracking and forecasting with rich maps},
    author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, De and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
    booktitle={Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={8748--8757},
    year={2019}
  }
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/dbdc27f7-cdf7-4509-bc52-ab2c7252d41b" alt="Argoverse 1" width="150">
</div>

> #### Dataset Overview
> -  324,557 interesting vehicle trajectories extracted from over 1,000 driving hours.
> - Geographic Info: Pittsburgh and Miami, USA
> - Agent Types: Cars, Pedestrians, Cyclists

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_argoverse.py`  
- **Config key:** `argoverse`  
- **Expected folder structure:**

```
argoverse
├── forecasting_train_v1.1
│   └── train
│       └── data
│           ├── 1.csv
│           └── ...
├── forecasting_val_v1.1
│   └── ...
├── forecasting_test_v1.1
│   └── ...
└── hd_map
    └── map_files
        ├── pruned_argoverse_MIA_10316_vector_map.xml
        ├── pruned_argoverse_PIT_10314_vector_map.xml
        └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_argoverse --config argoverse --path ../datasets
```

</details>

***

### *[Argoverse 2](https://arxiv.org/abs/2301.00493)*: Next Generation Datasets for Self-Driving Perception and Forecasting

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    We introduce Argoverse 2 (AV2) - a collection of three datasets for perception and forecasting research in the self-driving domain. The annotated Sensor Dataset contains 1,000 sequences of multimodal data, encompassing high-resolution imagery from seven ring cameras, and two stereo cameras in addition to lidar point clouds, and 6-DOF map-aligned pose. Sequences contain 3D cuboid annotations for 26 object categories, all of which are sufficiently-sampled to support training and evaluation of 3D perception models. The Lidar Dataset contains 20,000 sequences of unlabeled lidar point clouds and map-aligned pose. This dataset is the largest ever collection of lidar sensor data and supports self-supervised learning and the emerging task of point cloud forecasting. Finally, the Motion Forecasting Dataset contains 250,000 scenarios mined for interesting and challenging interactions between the autonomous vehicle and other actors in each local scene. Models are tasked with the prediction of future motion for "scored actors" in each scenario and are provided with track histories that capture object location, heading, velocity, and category. In all three datasets, each scenario contains its own HD Map with 3D lane and crosswalk geometry - sourced from data captured in six distinct cities. We believe these datasets will support new and existing machine learning research problems in ways that existing datasets do not. All datasets are released under the CC BY-NC-SA 4.0 license.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
 @inproceedings{Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}

```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/80fbda39-79e7-4acf-93b0-8965d894d14b" alt="Argoverse 2" width="200">
</div>


> #### Dataset Overview
> - 250,000 scenarios with trajectory data for many object types. This dataset improves upon the Argoverse 1 Motion Forecasting Dataset.
> - Austin, Detroit, Miami, Pittsburgh, Palo Alto, and Washington, D.C.
> - Mixed agent types including vehicles, pedestrians, and cyclists.

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_av2.py`  
- **Config key:** `av2`  
- **Expected folder structure:**

```
av2
├── train
│   ├── ...
│   └── ffffe3df-8d26-42c3-9e7a-59de044736a0
├── val
│   ├── ...
│   └── fffc6ef5-8fb4-4f20-afea-b9cb63c99182
└── test
    ├── ...
    └── fffc1965-9f9e-4822-ade7-750d87c4b7b9
```

- **Command:**

```bash
python -m preprocessing.preprocess_av2 --config av2 --path ../datasets
```

</details>

***

### *[Waymo Open Motion Dataset](https://arxiv.org/abs/2104.10133)*: Large Scale Interactive Motion Forecasting for Autonomous Driving 

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    As autonomous driving systems mature, motion forecasting has received increasing attention as a critical requirement for planning. Of particular importance are interactive situations such as merges, unprotected turns, etc., where predicting individual object motion is not sufficient. Joint predictions of multiple objects are required for effective route planning. There has been a critical need for high-quality motion data that is rich in both interactions and annotation to develop motion planning models. In this work, we introduce the most diverse interactive motion dataset to our knowledge, and provide specific labels for interacting objects suitable for developing joint prediction models. With over 100,000 scenes, each 20 seconds long at 10 Hz, our new dataset contains more than 570 hours of unique data over 1750 km of roadways. It was collected by mining for interesting interactions between vehicles, pedestrians, and cyclists across six cities within the United States. We use a high-accuracy 3D auto-labeling system to generate high quality 3D bounding boxes for each road agent, and provide corresponding high definition 3D maps for each scene. Furthermore, we introduce a new set of metrics that provides a comprehensive evaluation of both single agent and joint agent interaction motion forecasting models. Finally, we provide strong baseline models for individual-agent prediction and joint-prediction. We hope that this new large-scale interactive motion dataset will provide new opportunities for advancing motion forecasting models.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{ettinger2021large,
  title={Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset},
  author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R and Zhou, Yin and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9710--9719},
  year={2021}
}
```

</details>

<div align="center">  
  <img src=https://github.com/user-attachments/assets/ac50b87c-5d13-4510-906a-5519d94322c1 alt="Waymo logo" width="200">
</div>


> #### Dataset Overview
> - Composed of 103,354 segments each containing 20 seconds of object tracks at 10Hz and map data for the area covered by the segment.
> - Covers six cities in the United States: San Francisco, Mountain View, Los Angeles, Detroit, Seattle, and Phoenix.
> - Includes a diverse set of agent types such as vehicles, pedestrians, and cyclists.

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_waymo.py`  
- **Config key:** `waymo`  
- **Expected folder structure:**

```
waymo
├── training
│   ├── training.tfrecord-00000-of-01000
│   └── ...
├── validation
│   ├── validation.tfrecord-00000-of-00150
│   └── ...
└── testing
    ├── testing.tfrecord-00000-of-00150
    └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_waymo --config waymo --path ../datasets
```

</details>

***


### *[nuScenes](https://arxiv.org/abs/1903.11027)*: A Multimodal Dataset for Autonomous Driving

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Robust detection and tracking of objects is crucial for the deployment of autonomous vehicle technology. Image based benchmark datasets have driven development in computer vision tasks such as object detection, tracking and segmentation of agents in the environment. Most autonomous vehicles, however, carry a combination of cameras and range sensors such as lidar and radar. As machine learning based methods for detection and tracking become more prevalent, there is a need to train and evaluate such methods on datasets containing range sensor data along with images. In this work we present nuTonomy scenes (nuScenes), the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field of view. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bounding boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset. We define novel 3D detection and tracking metrics. We also provide careful dataset analysis as well as baselines for lidar and image based detection and tracking. Data, development kit and more information are available online.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{caesar2020nuscenes,
  title={{nuScenes: A} multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11621--11631},
  year={2020},
}
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/e4e4a9c8-edfd-4caf-83a4-d8531fc3a282" alt="motional" width="280">
</div>

> #### Dataset Overview
> - 1000 driving scenes, each 20 seconds long, with a full sensor suite including 6 cameras, 5 radars, and 1 lidar.
> - Geographic Info: Boston and Singapore
> - Agent Types: Cars, Pedestrians, Cyclists, Buses, Trucks, Motorcycles, and more.

</details>

<details>
<summary><strong>Preprocessing Instructions</strong></summary>

- **Script:** `preprocess_nuscenes.py`  
- **Config key:** `nuscenes`  
- **Expected folder structure:**

```
nuscenes
├── nuScenes-map-expansion-v1.3
│   ├── expansion
│   │   ├── boston-seaport.json
│   │   ├── singapore-onenorth.json
│   │   ├── singapore-hollandvillage.json
│   │   └── singapore-queenstown.json
│   └── ...
├── v1.0-trainval_meta
│   └── v1.0-trainval
│       ├── attribute.json
│       ├── ...
│       └── visibility.json
└── v1.0-test_meta
    └── v1.0-test
        ├── attribute.json
        ├── ...
        └── visibility.json
```

- **Command:**

```bash
python -m preprocessing.preprocess_nuscenes --config nuscenes --path ../datasets
```

</details>

***

### *[Lyft Level 5](https://arxiv.org/abs/2006.14480)*: Self-driving Motion Prediction Dataset


<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Motivated by the impact of large-scale datasets on ML systems we present the largest self-driving dataset for motion prediction to date, containing over 1,000 hours of data. This was collected by a fleet of 20 autonomous vehicles along a fixed route in Palo Alto, California, over a four-month period. It consists of 170,000 scenes, where each scene is 25 seconds long and captures the perception output of the self-driving system, which encodes the precise positions and motions of nearby vehicles, cyclists, and pedestrians over time. On top of this, the dataset contains a high-definition semantic map with 15,242 labelled elements and a high-definition aerial view over the area. We show that using a dataset of this size dramatically improves performance for key self-driving problems. Combined with the provided software kit, this collection forms the largest and most detailed dataset to date for the development of self-driving machine learning tasks, such as motion forecasting, motion planning and simulation.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{houston2021one,
  title={One thousand and one hours: {S}elf-driving motion prediction dataset},
  author={Houston, John and Zuidhof, Guido and Bergamini, Luca and Ye, Yawei and Chen, Long and Jain, Ashesh and Omari, Sammy and Iglovikov, Vladimir and Ondruska, Peter},
  booktitle={Conference on Robot Learning (CoRL)},
  pages={409--418},
  year={2021},
  organization={PMLR}
}
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/cacd6d39-1c10-477a-99d9-1d560ac7dad4" alt="Lyft logo" width="140">
</div>


> #### Dataset Overview
> - 170,000 scenes, each 25 seconds long, accompanied by HD semantic maps.
> - Geographic Info: Palo Alto, California, USA
> - Agent Types: Cars, Pedestrians, Cyclists

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

The **Lyft Level 5** dataset stores trajectory data in a `zarr` format.
To process this dataset, the preprocessing script relies on the `zarr` Python library for reading and parsing the trajectory arrays.
This is an optional dependency, so you may need to install it separately if you haven't done so already.

- **Script:** `preprocess_lyft.py`  
- **Config key:** `lyft`  
- **Expected folder structure:**

```
lyft
├── semantic_map
│   ├── semantic_map.pb
│   └── ...
├── train
│   └── train.zarr
│   └── ...
└── validate
    └── validate.zarr
    └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_lyft --config lyft --path ../datasets
```

</details>

***

### *[View-of-Delft Prediction](https://pure.tudelft.nl/ws/portalfiles/portal/190220102/Multi-Class_Trajectory_Prediction_in_Urban_Traffic_Using_the_View-of-Delft_Prediction_Dataset.pdf)*: Multi-Class Trajectory Prediction in Urban Traffic


<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    This letter presents View-of-Delft Prediction, a new dataset for trajectory prediction, to address the lack of on-board trajectory datasets in urban mixed-traffic environments. View-of-Delft Prediction builds on the recently released urban View-of-Delft (VoD) dataset to make it suitable for trajectory prediction. Unique features of this dataset are the challenging road layouts of Delft, with many narrow roads and bridges, and the close proximity between vehicles and Vulnerable Road Users (VRUs). It contains a large proportion of VRUs, with 569 prediction instances for vehicles, 347 for cyclists, and 934 for pedestrians. We additionally provide high-definition map annotations for the VoD dataset to enable state-of-the-art prediction models to be used. We analyse two state-of-the-art trajectory prediction models, PGP and P2T, which originally were developed for vehicle-dominated traffic scenarios, to assess the strengths and weaknesses of current modelling approaches in mixed traffic settings with large numbers of VRUs. Our analysis shows that there is a significant domain gap between the vehicle-dominated nuScenes and VRU-dominated VoD Prediction datasets. The dataset is publicly released for non-commercial research purposes.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@article{boekema2024vodp,
  author={Boekema, Hidde J-H. and Martens, Bruno K.W. and Kooij, Julian F.P. and Gavrila, Dariu M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-class Trajectory Prediction in Urban Traffic using the View-of-Delft Prediction Dataset}, 
  year={2024},
  volume={9},
  number={5},
  pages={4806-4813},
  keywords={Trajectory;Roads;Annotations;Semantics;Pedestrians;Predictive models;History;Datasets for Human Motion;Data Sets for Robot Learning;Deep Learning Methods},
  doi={10.1109/LRA.2024.3385693}}
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/39c44e43-0896-4b93-bea5-e719b67a82a0" alt="TUDelft logo" width="180">
</div>


> #### Dataset Overview
> - 1850 scenes, accompanied by HD semantic maps.
> - Geographic Info: Delft, Netherlands
> - Agent Types: Vehicles, Pedestrians, Cyclists

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_vod.py`  
- **Config key:** `vod`  
- **Expected folder structure:**

```
vod
├── maps
│   └── expansion
│       └── delft.json
├── v1.0-trainval
│   ├── attribute.json
│   ├── ...
│   └── sensor.json
└── v1.0-test
    ├── attribute.json
    ├── ...
    └── sensor.json
```


- **Command:**

```bash
python -m preprocessing.preprocess_vod --config vod --path ../datasets
```

</details>


***


### *[ApolloScape](https://arxiv.org/pdf/1811.02146)*: Trajectory Prediction for Heterogeneous Traffic-Agents


<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        To safely and efficiently navigate in complex urban traffic, autonomous vehicles must make responsible predictions in relation to surrounding traffic-agents (vehicles, bicycles, pedestrians, etc.). A challenging and critical task is to explore the
        movement patterns of different traffic-agents and predict their
        future trajectories accurately to help the autonomous vehicle
        make reasonable navigation decision. To solve this problem,
        we propose a long short-term memory-based (LSTM-based)
        realtime traffic prediction algorithm, TrafficPredict. Our approach uses an instance layer to learn instances' movements
        and interactions and has a category layer to learn the similarities of instances belonging to the same type to refine the
        prediction. In order to evaluate its performance, we collected
        trajectory datasets in a large city consisting of varying conditions and traffic densities. The dataset includes many challenging scenarios where vehicles, bicycles, and pedestrians
        move among one another. We evaluate the performance of
        TrafficPredict on our new dataset and highlight its higher accuracy for trajectory prediction by comparing with prior prediction methods.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{ma2019trafficpredict,
      title={Trafficpredict: Trajectory prediction for heterogeneous traffic-agents},
      author={Ma, Yuexin and Zhu, Xinge and Zhang, Sibo and Yang, Ruigang and Wang, Wenping and Manocha, Dinesh},
      booktitle={Proceedings of the AAAI conference on artificial intelligence},
      volume={33},
      number={01},
      pages={6120--6127},
      year={2019}
    }
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/7bdb09b8-0888-4984-8824-fcf55bfb6f52" alt="ApolloScape logo" width="250">
</div>

> #### Dataset Overview
> - The trajectory dataset consists of 53min training sequences and 50min testing sequences captured at 2 frames per second.
> - Multiple urban driving scenarios captured using an instrumented vehicle
> - In total ~ 82,000 agents
> - Road user classes: vehicle, pedestrian, motorcycle/bicycle

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_apollo.py`  
- **Config key:** `apollo`  
- **Expected folder structure:**

```
apollo
├── prediction_train
│   ├── result_9048_1.frame.txt
│   ├── result_9048_3.frame.txt
│   └── ...
└── prediction_test
    └── prediction_test.txt
```

- **Command:**

```bash
python -m preprocessing.preprocess_apollo --config apollo --path ../datasets
```

</details>

***

### *[INTERACTION](https://arxiv.org/abs/1910.03088)*: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Interactive motion datasets of road participants are
        vital to the development of autonomous vehicles in both industry
        and academia. Research areas such as motion prediction, motion
        planning, representation learning, imitation learning, behavior
        modeling, behavior generation, and algorithm testing, require
        support from high-quality motion datasets containing interactive
        driving scenarios with different driving cultures. In this paper,
        we present an INTERnational, Adversarial and Cooperative
        moTION dataset (INTERACTION dataset) in interactive driving
        scenarios with semantic maps.
        Five features of the dataset are highlighted. 1) The interactive
        driving scenarios are diverse, including urban/highway/ramp
        merging and lane changes, roundabouts with yield/stop signs,
        signalized intersections, intersections with one/two/all-way stops, etc.
        2) Motion data from different countries and different continents
        are collected so that driving preferences and styles in different
        cultures are naturally included. 3) The driving behavior is highly
        interactive and complex with adversarial and cooperative motions
        of various traffic participants. Highly complex behavior such
        as negotiations, aggressive/irrational decisions and traffic rule
        violations are densely contained in the dataset, while regular
        behavior can also be found from cautious car-following, stop,
        left/right/U-turn to rational lane-change and cycling and pedestrian crossing,
        etc. 4) The levels of criticality span wide, from
        regular safe operations to dangerous, near-collision maneuvers.
        Real collision, although relatively slight, is also included. 5) Maps
        with complete semantic information are provided with physical
        layers, reference lines, lanelet connections and traffic rules.
        The data is recorded from drones and traffic cameras, and the
        processing pipelines for both are briefly described. Statistics of
        the dataset in terms of number of entities and interaction density
        are also provided, along with some utilization examples in the
        areas of motion prediction, imitation learning, decision-making
        and planing, representation learning, interaction extraction and
        social behavior generation. The dataset can be downloaded via
        https://interaction-dataset.com.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @article{interactiondataset,
    title = {{INTERACTION} {Dataset}: {An} {INTERnational}, {Adversarial} and {Cooperative} {moTION} {Dataset} in {Interactive} {Driving} {Scenarios} with {Semantic} {Maps}},
    journal={arXiv preprint arXiv:1910.03088},
    author = {Zhan, Wei and Sun, Liting and Wang, Di and Shi, Haojie and Clausse, Aubrey and Naumann, Maximilian and K\"ummerle, Julius and K\"onigshof, Hendrik and Stiller, Christoph and de La Fortelle, Arnaud and Tomizuka, Masayoshi},
    year = {2019}}
```

</details>

<div align="center">  
  <img src=https://github.com/user-attachments/assets/bfb39462-b124-4942-bd89-7312cda6f37b alt="INTERACTION.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset from 18 different recording locations across three continents
> - Multiple driving scenarios: urban/highway/ramp merging, roundabouts, intersections
> - In total ~ 40,000 vehicles
> - Road user classes: car, pedestrian/bicycle

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_interact.py`  
- **Config key:** `interact`  
- **Expected folder structure:**

```
interact
├── maps
│   ├── DR_CHN_Merging_ZS0.osm
│   └── ...
├── test_conditional-multi-agent
│   ├── DR_CHN_Merging_ZS0_obs.csv
│   └── ...
├── test_multi-agent
│   ├── DR_CHN_Merging_ZS0_obs.csv
│   └── ...
├── train
│   ├── DR_CHN_Merging_ZS0_obs.csv
│   └── ...
└── val
    ├── DR_CHN_Merging_ZS0_obs.csv
    └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_interact --config interact --path ../datasets
```

</details>

***

### *[highD](https://arxiv.org/abs/1810.05642)*: The Highway Drone Dataset

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Scenario-based testing for the safety validation of
        highly automated vehicles is a promising approach that is being
        examined in research and industry. This approach heavily relies
        on data from real-world scenarios to derive the necessary
        scenario information for testing. Measurement data should be
        collected at a reasonable effort, contain naturalistic behavior of
        road users and include all data relevant for a description of the
        identified scenarios in sufficient quality. However, the current
        measurement methods fail to meet at least one of the
        requirements. Thus, we propose a novel method to measure data
        from an aerial perspective for scenario-based validation
        fulfilling the mentioned requirements. Furthermore, we provide
        a large-scale naturalistic vehicle trajectory dataset from German
        highways called highD. We evaluate the data in terms of
        quantity, variety and contained scenarios. Our dataset consists
        of 16.5 hours of measurements from six locations with 110 000
        vehicles, a total driven distance of 45 000 km and 5600 recorded
        complete lane changes.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{highDdataset,
       title={The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems},
       author={Krajewski, Robert and Bock, Julian and Kloeker, Laurent and Eckstein, Lutz},
       booktitle={2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
       pages={2118-2125},
       year={2018},
       doi={10.1109/ITSC.2018.8569552}
    }
```

</details>

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/0e9de880-9ee3-4941-ab41-692f259a0cbc alt="highD.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset on six different recording locations
> - In total ~ 110,500 vehicles
> - Road user classes: car, trucks

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_highway.py`  
- **Config key:** `highD`  
- **Expected folder structure:**

```
highD
└── data
    ├── 01_recordingMeta.csv
    ├── 01_tracks.csv
    ├── 01_tracksMeta.csv
    └── ...
```
- **Command:**

```bash
python -m preprocessing.preprocess_highway --config highD --path ../datasets
```
</details>


***

### *[rounD](https://ieeexplore.ieee.org/document/9294728)*: The Roundabouts Drone Dataset

<details>
  <summary><strong>Dataset details</strong></summary>


<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        The development and validation of automated vehicles involves a large number of challenges to be overcome.
        Due to the high complexity, many classic approaches quickly reach their limits and data-driven methods become necessary.
        This creates an unavoidable need for trajectory datasets of road users in all relevant traffic scenarios.
        As these trajectories should include naturalistic and diverse behavior, they have to be recorded in public traffic.
        Roundabouts are particularly interesting because of the density of interaction between road users, which must be considered by an automated vehicle for behavior planning.
        We present a new dataset of road user trajectories at roundabouts in Germany.
        Using a camera-equipped drone, traffic at a total of three different roundabouts in Germany was recorded.
        The tracks consisting of positions, headings, speeds, accelerations and classes of objects were extracted from recorded videos using deep neural networks.
        The dataset contains a total of six hours of recordings with more than 13 746 road users including cars, vans, trucks, buses, pedestrians, bicycles and motorcycles.
        In order to make the dataset as accessible as possible for tasks like scenario classification, road user behavior prediction or driver modeling, we provide source code for parsing and visualizing the dataset as well as maps of the recording sites.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{rounDdataset,
        title={The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany},
        author={Krajewski, Robert and Moers, Tobias and Bock, Julian and Vater, Lennart and Eckstein, Lutz},
        booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)},
        pages={1-6},
        year={2020},
        doi={10.1109/ITSC45102.2020.9294728}
    }
```

</details>

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/89b37a52-9b78-42a6-9386-0b2d5b5caf33 alt="rounD.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset on three different recording locations
> - In total ~ 13,740 road users
> - Road user classes: car, trailer, truck, bus, motorcycle, bicycle, pedestrian

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_urban.py`  
- **Config key:** `rounD`  
- **Expected folder structure:**

```
rounD
├── data
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps
  └── lanelets
    ├── 0_neuweiler
    └── ...

```

- **Command:**
```bash
python -m preprocessing.preprocess_urban --config rounD --path ../datasets
```
</details>


***

### *[inD](https://arxiv.org/abs/1911.07602)*: The Intersections Drone Dataset

<details>
  <summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Automated vehicles rely heavily on data-driven
        methods, especially for complex urban environments. Large
        datasets of real world measurement data in the form of road
        user trajectories are crucial for several tasks like road user
        prediction models or scenario-based safety validation. So far,
        though, this demand is unmet as no public dataset of urban
        road user trajectories is available in an appropriate size, quality
        and variety. By contrast, the highway drone dataset (highD) has
        recently shown that drones are an efficient method for acquiring
        naturalistic road user trajectories. Compared to driving studies
        or ground-level infrastructure sensors, one major advantage of
        using a drone is the possibility to record naturalistic behavior,
        as road users do not notice measurements taking place. Due to
        the ideal viewing angle, an entire intersection scenario can be
        measured with significantly less occlusion than with sensors at
        ground level. Both the class and the trajectory of each road
        user can be extracted from the video recordings with high
        precision using state-of-the-art deep neural networks. Therefore,
        we propose the creation of a comprehensive, large-scale urban
        intersection dataset with naturalistic road user behavior using
        camera-equipped drones as successor of the highD dataset. The
        resulting dataset contains more than 11500 road users including
        vehicles, bicyclists and pedestrians at intersections in Germany
        and is called inD. The dataset consists of 10 hours of measurement
        data from four intersections.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{inDdataset,
        title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
        author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
        booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
        pages={1929-1934},
        year={2020},
        doi={10.1109/IV47402.2020.9304839}
    }
```

</details>

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/98c48e3a-8ac8-4896-863c-c26e08d6764b alt="inD.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset on four different recording locations
> - In total ~ 8,200 vehicles and ~ 5,300 vulnerable road users (VRUs)
> - Road user classes: car, truck/bus, bicycle, pedestrian

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_urban.py`  
- **Config key:** `inD`  
- **Expected folder structure:**

```
inD
├── data
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps
  └── ...

```

- **Command:**
```bash
python -m preprocessing.preprocess_urban --config inD --path ../datasets
```
</details>

***

### *[exiD](https://ieeexplore.ieee.org/document/9827305)*: The Entries and Exits Drone Dataset


<details>
  <summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Development and safety validation of highly automated vehicles increasingly relies on data and data-driven methods. In processing sensor datasets for environment perception, it is common to use public and commercial datasets for training and evaluating machine learning based systems. For system-level evaluation and safety validation of an automated driving system, real-world trajectory datasets are of great value for several tasks in the process, i.a. for testing in simulation, scenario extraction or training of road user agent models. Ground-based recording methods such as sensor-equipped vehicles or infrastructure sensors are sometimes limited, for instance, due to their field of view. Camera-equipped drones, however, offer the ability to record road users without vehicle-to-vehicle occlusion and without influencing traffic. The highway drone dataset (highD) has shown that the recording method is efficient in terms of cumulative kilometers and has become a benchmark dataset for many research questions. It contains many vehicle interactions due to dense traffic, but lacks merging scenarios, which are challenging for highly automated vehicles. Therefore, we propose this highway drone dataset called exiD, recorded using camera-equipped drones at entries and exits on the German Autobahn. The dataset contains 69 172 road users classified as car, truck and vans and a total amount of more than 16 hours of measurement data.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{exiDdataset,
        title={The exiD Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany},
        author={Moers, Tobias and Vater, Lennart and Krajewski, Robert and Bock, Julian and Zlocki, Adrian and Eckstein, Lutz},
        booktitle={2022 IEEE Intelligent Vehicles Symposium (IV)},
        pages={958-964},
        year={2022},
        doi={10.1109/IV51971.2022.9827305}
    }
```

</details>

<div align="center">
  <img src=https://github.com/user-attachments/assets/69a75f22-2cdf-414d-8238-aefe246f7803 alt="exiD.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset on seven different recording locations
> - In total ~ 69,430 road users
> - Road user classes: car, truck, bus, motorcycle

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_highway.py`  
- **Config key:** `exiD`  
- **Expected folder structure:**

```
exiD
├── data
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps
  └── ...

```

- **Command:**
```bash
python -m preprocessing.preprocess_highway --config exiD --path ../datasets
```
</details>

***

### *[uniD](https://levelxdata.com/unid-dataset/)*: The University Drone Dataset

<details>
  <summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        The uniD dataset is an innovative collection of naturalistic road user trajectories, captured within the RWTH Aachen University campus using drone technology to address common challenges such as occlusions found in traditional traffic data collection methods. It meticulously documents the movement and classifies each road user by type. Employing cutting-edge computer vision algorithms, the dataset ensures high positional accuracy. Its utility spans various applications, from predicting road user behavior and modeling driver actions to conducting scenario-based safety checks for automated driving systems and facilitating the data-driven design of Highly Automated Driving (HAD) system components.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @misc{uniDdataset,
      title = {{The uniD Dataset: A} university drone dataset},
      author = {{leveLXData}},
      year = {2024},
      howpublished = {\url{https://levelxdata.com/unid-dataset/}},
      note = {Accessed: ...}
    }
```

</details>

<div align="center">
  <img src=https://github.com/user-attachments/assets/ac11c28c-4d25-4d06-8f6d-90d0140065df alt="uniD.gif">
</div>

> #### Dataset Overview
> - Naturalistic trajectory dataset on one recording location
> - In total ~ 1,380 vehicles and ~ 8,600 vulnerable road users (VRUs)
> - All road users classes: car, truck/bus, bicycle, pedestrian

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_urban.py`  
- **Config key:** `uniD`  
- **Expected folder structure:**

```
uniD
├── data
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps
  └── ...

```

- **Command:**
```bash
python -m preprocessing.preprocess_urban --config uniD --path ../datasets
```
</details>


***

### *[openDD](https://arxiv.org/abs/2007.08463)*: A Large-Scale Roundabout Drone Dataset

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Analyzing and predicting the traffic scene around the ego vehicle has been one of the key challenges in autonomous driving. Datasets including the trajectories of all road users present in a scene, as well as the underlying road topology are invaluable to analyze the behavior of the different traffic participants. The interaction between the various traffic participants is especially high in intersection types that are not regulated by traffic lights, the most common one being the roundabout.
    We introduce the openDD dataset, including 84,774 accurately tracked trajectories and HD map data of seven different roundabouts. The openDD dataset is annotated using images taken by a drone in 501 separate flights, totalling in over 62 hours of trajectory data. As of today, openDD is by far the largest publicly available trajectory dataset recorded from a drone perspective, while comparable datasets span 17 hours at most.
    The data is available, for both commercial and noncommercial use, at: https://l3pilot.eu/data/opendd.html 
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{breuer2020opendd,
  title={{openDD: A} large-scale roundabout drone dataset},
  author={Breuer, Antonia and Term{\"o}hlen, Jan-Aike and Homoceanu, Silviu and Fingscheidt, Tim},
  booktitle={IEEE Intelligent Transportation Systems Conference (ITSC)},
  pages={1--6},
  year={2020},
}
```

</details>

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0c66cbe7-0165-4087-a804-698a701e6187" alt="Roundabout 1"/></td>
    <td><img src="https://github.com/user-attachments/assets/93fc0058-e636-44db-a1b9-f9939a124bb5" alt="Roundabout 3"/></td>
    <td><img src="https://github.com/user-attachments/assets/e1871afa-0663-4430-8ea4-2f0d4255c936" alt="Roundabout 7"/></td>
  </tr>
</table>


> #### Dataset Overview
> - 84,774 trajectories recorded from a drone perspective at seven different roundabouts.
> - Geographic Info: Various roundabouts in Germany
> - Agent Types: Cars, Bicycles, Pedestrians, Trucks, Buses


</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_opendd.py`  
- **Config key:** `opendd`  
- **Expected folder structure:**

```
openDD
├── opendd_v3-rdb1
│   ├── rdb1
│   │   └── map_rdb1
│   │       ├── ...
│   │       └── map_rdb1.sqlite
│   ├── trajectories_rdb1_v3.sqlite
│   └── ...
├── opendd_v3-rdb2
│   ├── rdb2
│   │   └── map_rdb2
│   │       ├── ...
│   │       └── map_rdb2.sqlite
│   ├── trajectories_rdb2_v3.sqlite
│   └── ...
:
└── opendd_v3-rdb7
    ├── rdb7
    │   └── map_rdb7
    │       ├── ...
    │       └── map_rdb7.sqlite
    ├── trajectories_rdb7_v3.sqlite
    └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_opendd --config opendd --path ../datasets
```

</details>

***

### *[AD4CHE](https://ieeexplore.ieee.org/document/10079130)*: Aerial Dataset for China's Congested Highways & Expressways

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Autonomous driving has attracted considerable attention from research and industry communities. Although prototypes of automated vehicles (AVs) are developed, remaining safety issues and functional insufficiencies hinder their market introduction. To obtain reasonably foreseeable scenarios and study human driving policies, many naturalistic driving datasets are proposed. However, no open-source dataset filled with congestion scenarios is publicly available. The paper presents the Aerial Dataset for China's Congested Highways & Expressways (AD4CHE). It contains 5.12 hours of aerial survey data from four different cities in China, with a total driving distance of 6540.7 km. Moreover, overlap and non-overlap cut-in scenarios are distinguished to better describe driver behavior in congestion scenarios. Both types of cut-in scenarios are extracted and parameterized. The Kernel Density Estimator (KDE) is utilized to generate parameter distributions for the scenario-based testing method. Furthermore, the driving behavior in overlap cut-in scenarios is intensively analyzed. The results reveal that the drivers have an evasive maneuver during overlap cut-in of challenging vehicles, and the preferred following distance varies with the relative longitudinal velocity. Both scenario parameterization and driving behavior analysis can contribute to developing and verifying Traffic Jam Pilot (TJP) systems deployed in Chinese traffic situations.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@article{zhang2023ad4che,
  title={The {AD4CHE} dataset and its application in typical congestion scenarios of traffic jam pilot systems},
  author={Zhang, Yuxin and Wang, Cheng and Yu, Ruilin and Wang, Luyao and Quan, Wei and Gao, Yang and Li, Pengfei},
  journal={IEEE Transactions on Intelligent Vehicles},
  volume={8},
  number={5},
  pages={3312--3323},
  year={2023},
  publisher={IEEE}
}
```

</details>

<div align="center">  
  <img src="https://github.com/user-attachments/assets/f54b7f20-57df-4675-b6d1-8e0b53f93a79" alt="AD4CHE", width="400">
</div>

> #### Dataset Overview
> - 5.12 hours of aerial survey data from four different cities in China, covering a total driving distance of 6540.7 km.
> - Geographic Info: Various congested highways and expressways in China
> - Agent Types: Cars, Trucks, Buses, Motorcycles

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>
Unlike most datasets supported in this repository, AD4CHE provides its maps as raw images rather than structured lane graph data.  
To address this, a custom preprocessing pipeline has been developed using OpenCV to extract approximate lane information and convert it into a simplified polyline format.
This is an optional dependency, so you may need to install it separately if you haven't done so already.

- **Script:** `preprocess_highway.py`  
- **Config key:** `ad4che`  
- **Expected folder structure:**

```
ad4che
├── AD4CHE_Data_v1.0
│   ├── DJI_0001
│   │   ├── 01_lanePicture.png
│   │   ├── 01_recordingMeta.csv
│   │   ├── 01_tracksMeta.csv
│   │   ├── 01_tracks.csv
│   │   └── ...
│   ├── DJI_0002
│   │   ├── 02_lanePicture.png
│   │   ├── 02_recordingMeta.csv
│   │   ├── 02_tracksMeta.csv
│   │   ├── 02_tracks.csv
│   │   └── ...
│   :
│   └── DJI_0068
│       └── ...
└── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_highway --config ad4che --path ../datasets
```

</details>


***

### *[SIND](https://arxiv.org/abs/2209.02297)*: A Drone Dataset at Signalized Intersection in China

<details>
  <summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Intersection is one of the most challenging scenarios for autonomous driving tasks. Due to the complexity and stochasticity, essential applications (e.g., behavior modeling, motion prediction, safety validation, etc.) at intersections rely heavily on data-driven techniques. Thus, there is an intense demand for trajectory datasets of traffic participants (TPs) in intersections. Currently, most intersections in urban areas are equipped with traffic lights. However, there is not yet a large-scale, high-quality, publicly available trajectory dataset for signalized intersections. Therefore, in this paper, a typical two-phase signalized intersection is selected in Tianjin, China. Besides, a pipeline is designed to construct a Signalized INtersection Dataset (SIND), which contains 7 hours of recording including over 13,000 TPs with 7 types. Then, the behaviors of traffic light violations in SIND are recorded. Furthermore, the SIND is also compared with other similar works. The features of the SIND can be summarized as follows: 1) SIND provides more comprehensive information, including traffic light states, motion parameters, High Definition (HD) map, etc. 2) The category of TPs is diverse and characteristic, where the proportion of vulnerable road users (VRUs) is up to 62.6% 3) Multiple traffic light violations of non-motor vehicles are shown. We believe that SIND would be an effective supplement to existing datasets and can promote related research on autonomous driving.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @inproceedings{xu2022drone,
      title={{SIND: A} drone dataset at signalized intersection in China},
      author={Xu, Yanchao and Shao, Wenbo and Li, Jun and Yang, Kai and Wang, Weida and Huang, Hua and Lv, Chen and Wang, Hong},
      booktitle={25th International Conference on Intelligent Transportation Systems (ITSC)},
      pages={2471--2478},
      year={2022},
      organization={IEEE}
    }
```

</details>

<table>
  <tr>
    <td><img src=https://github.com/user-attachments/assets/d458a2bb-2443-421c-8b54-49a34535651a alt="Chongqing_NR"/></td>
    <td><img src=https://github.com/user-attachments/assets/477693df-3e85-42e0-a317-e69ac16dcc09 alt="Changchun_Pudong"/></td>
    <td><img src=https://github.com/user-attachments/assets/6facdf9f-dbfa-412c-861a-3eda9bb4e4ed alt="Xi'an_Shanglin"/></td>
  </tr>
</table>

> #### Dataset Overview
> - Naturalistic trajectory dataset on four different recording locations
> - Over 13,000 traffic participants of various types
> - Traffic light states included
> - Road user classes: car, truck, bus, motorcycle, tricycle, bicycle, pedestrian


<br>

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

We provide two options for using the SIND dataset: 1) using the entire dataset, or 2) using the demo dataset available
on the [SIND GitHub repository](www.github.com/SOTIF-AVLab/SinD).
Using the demo dataset is recommended for users who want to quickly test the toolbox.
Its use is straightforward, simply name the repository root folder `SIND_demo`, place it in the `datasets` directory,
and run the preprocessing script with the `demo` configuration.
> Note that some files in the repo are quite large and may require git-lfs to download properly.

For users who want to use the entire dataset, we ask you to organize the data to match the structure of the other
datasets (see below).

- **Script:** `preprocess_urban.py`  
- **Config key:** `SIND`  
- **Expected folder structure:**

```
SIND
├── data
│   ├── 6_22_NR_1
│   ├── 6_22_NR_2
│   ├── ...
│   ├── xian_415_n2
│   └── xian_415_n5
└── maps
    ├── Changchun_Pudong.osm
    ├── map_relink_law_save.osm
    ├── NR_ll2.osm
    └── Xi'an_Shanglin.osm
```
> Note that `map_relink_law_save.osm` needs to be downloaded from the GitHub repository.

- **Command:**
```bash
python -m preprocessing.preprocess_urban --config SIND --path ../datasets
```
</details>


***

### *[A43](https://data.isac.rwth-aachen.de/?p=58)*: Vehicle Trajectory Dataset from Drone Videos Including Off-Ramp and Congested Traffic

<details>
  <summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Vehicle trajectory data have become essential for many research fields, such as traffic flow, traffic safety and automated driving. In order to make trajectory data usable for researchers, an overview of the included road section and traffic situation as well as a description of the data processing methodology is necessary. In this paper, we present a trajectory dataset from a German highway with two lanes per direction, an off-ramp and congested traffic in one direction, and an on-ramp in the other direction. The dataset contains 8,648 trajectories and covers 87 minutes and a ~1,200 m long section of the road. The trajectories were extracted from drone videos using a post-trained yolov5 object detection model and projected onto the road surface using a 3D camera calibration. The post-processing methodology can compensate for most false detections and yield accurate speeds and accelerations. We present some applications of the data including a traffic flow analysis and accident risk analysis. The trajectory data are also compared with induction loop data and vehicle-based smartphone sensor data in order to evaluate the plausibility and quality of the trajectory data. The deviations of the speeds and accelerations are estimated at 0.45 m/s and 0.3 m/s2 respectively.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @article{berghaus2024vehicle,
      title={Vehicle trajectory dataset from drone videos including off-ramp and congested traffic--Analysis of data quality, traffic flow, and accident risk},
      author={Berghaus, Moritz and Lamberty, Serge and Ehlers, J{\"o}rg and Kall{\'o}, Eszter and Oeser, Markus},
      journal={Communications in Transportation Research},
      volume={4},
      pages={100133},
      year={2024},
      publisher={Elsevier}
    }
```

</details>

<table>
  <tr>
    <td><img src=https://github.com/user-attachments/assets/a91f70de-10d5-4ae1-9e13-87d5d2ecf9a5 alt="East"/></td>
  </tr>
    <td><img src=https://github.com/user-attachments/assets/5596b220-7943-44e0-971c-3cb6b2f6d987 alt="West"/></td>
</table>

> #### Dataset Overview
> - Naturalistic trajectory dataset from highway A43 near Münster, Germany
> - In total ~ 8,600 vehicles
> - All road users classes: car, truck, bus, motorcycle

<br>

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_highway.py`  
- **Config key:** `A43`  
- **Expected folder structure:**

```
A43
├── DroneDataEastToWestCSV_220725.csv
├── DroneDataWestToEastCSV_220725.csv
└── ...

```

- **Command:**
```bash
python -m preprocessing.preprocess_highway --config A43 --path ../datasets
```
</details>

***


### *[I-80](https://www.fhwa.dot.gov/publications/research/operations/06137/)*: Interstate 80 Freeway Dataset

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    To support the development of microscopic driver behavior algorithms, the Next Generation SIMulation (NGSIM) program is collecting detailed, high-quality traffic datasets. NGSIM stakeholder groups identified the collection of real-world vehicle trajectory data as important to understanding and researching microscopic driver behavior. The NGSIM datasets represent the most detailed and accurate field data collected to date for traffic microsimulation research and development. The Interstate 80 (I-80) freeway dataset was the first of several datasets collected under the NGSIM program.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@misc{i80dataset,
  author = {{U.S. Department of Transportation Federal Highway Administration}},
  year = {2016},
  title = {Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data. I-80.},
  howpublished = {\url{http://doi.org/10.21949/1504477}},
  note = {Dataset provided by ITS DataHub through Data.transportation.gov. Accessed YYYY-MM-DD}
}
```

</details>

<div align="center">  
  <img src=https://github.com/user-attachments/assets/cf8dd343-992f-4c34-be6e-daffa7b4389a alt="I-80", width="300">
</div>

> #### Dataset Overview
> - 45 minutes of traffic data collected on the I-80 freeway in California, segmented into 15-minute intervals.
> - Geographic Info: I-80 Freeway, California, USA
> - Agent Types: Cars, Trucks, Motorcycles

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_highway.py`  
- **Config key:** `i80`  
- **Expected folder structure:**

```
i80
├── 0400pm-0415pm
│   ├── trajectories-0400-0415.csv
│   └── ...
├── 0500pm-0515pm
│   ├── trajectories-0500-0515.csv
│   └── ...
└── 0515pm-0530pm
    ├── trajectories-0515-0530.csv
    └── ...

```

- **Command:**

```bash
python -m preprocessing.preprocess_highway --config i80 --path ../datasets
```

</details>


***

### *[US-101](https://www.fhwa.dot.gov/publications/research/operations/07030/)*: US Highway 101 Dataset

<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    To support the development of algorithms for driver behavior at microscopic levels, the Next Generation SIMulation (NGSIM) computer program is collecting detailed, high-quality traffic datasets. NGSIM stakeholder groups identified the collection of real-world vehicle trajectory data as important to understanding and researching driver behavior at microscopic levels. The NGSIM datasets represent the most detailed and accurate field data collected to date for traffic microsimulation research and development. The US Highway 101 (US 101) dataset was one of several datasets collected under the NGSIM program.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@misc{us101dataset,
  author = {{U.S. Department of Transportation Federal Highway Administration}},
  year = {2016},
  title = {Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data. US-101.},
  howpublished = {\url{http://doi.org/10.21949/1504477}},
  note = {Dataset provided by ITS DataHub through Data.transportation.gov. Accessed YYYY-MM-DD}
}
```

</details>

<div align="center">  
  <img src=https://github.com/user-attachments/assets/b7813a2e-ad6d-4369-b7d9-fe217cf0b5ce alt="US-101", width="300">
</div>

> #### Dataset Overview
> - 45 minutes of traffic data collected on US Highway 101 in California, segmented into 15-minute intervals.
> - Geographic Info: US Highway 101, California, USA
> - Agent Types: Cars, Trucks, Motorcycles

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

- **Script:** `preprocess_highway.py`  
- **Config key:** `us101`  
- **Expected folder structure:**

```
us101
├── 0750am-0805am
│   ├── trajectories-0750am-0805am.csv
│   └── ...
├── 0805am-0820am
│   ├── trajectories-0805am-0820am.csv
│   └── ...
└── 0820am-0835am
    ├── trajectories-0820am-0835am.csv
    └── ...
```

- **Command:**

```bash
python -m preprocessing.preprocess_highway --config us101 --path ../datasets
```

</details>


***

### *[ETH](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5459260)*: You'll Never Walk Alone


<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Object tracking typically relies on a dynamic model to
predict the object’s location from its past trajectory. In
crowded scenarios a strong dynamic model is particularly
important, because more accurate predictions allow for
smaller search regions, which greatly simplifies data association. Traditional dynamic models predict the location
for each target solely based on its own history, without taking into account the remaining scene objects. Collisions
are resolved only when they happen. Such an approach
ignores important aspects of human behavior: people are
driven by their future destination, take into account their
environment, anticipate collisions, and adjust their trajectories at an early stage in order to avoid them. In this work,
we introduce a model of dynamic social behavior, inspired
by models developed for crowd simulation. The model is
trained with videos recorded from birds-eye view at busy
locations, and applied as a motion model for multi-people
tracking from a vehicle-mounted camera. Experiments on
real sequences show that accounting for social interactions
and scene knowledge improves tracking performance, especially during occlusions.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{pellegrini2009you,
  title={{You'll never walk alone: M}odeling social behavior for multi-target tracking},
  author={Pellegrini, Stefano and Ess, Andreas and Schindler, Konrad and Van Gool, Luc},
  booktitle={International Conference on Computer Vision (ICCV)},
  pages={261--268},
  year={2009},
  organization={IEEE}
}
```

</details>

> #### Dataset Overview
> - The ETH dataset contains pedestrian trajectories recorded in urban environments.
> - Scene types include busy streets and public squares.

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

For the ETH and UCY datasets, we use the widely adopted train-val-test split and adopt the leave-one-out strategy from the Social-LSTM (Social-GAN) paper. The split can be downloaded here: [ETH-UCY Split](https://www.dropbox.com/s/8n02xqv3l9q18r1).

- **Script:** `preprocess_ethucy.py`  
- **Config key:** `ethucy`  
- **Expected folder structure:**

```
ethucy
├── eth
│   ├── train
│   │   ├── biwi_hotel_train.txt
│   │   ├── ...
│   │   └── uni_examples_train.txt
│   ├── val
│   │   └── ...
│   └── test
│       └── ...
├── hotel
│   └── ...
├── univ
│   └── ...
├── zara1
│   └── ...
└── zara2
    └── ...

```

- **Command:**

```bash
python -m preprocessing.preprocess_ethucy --config ethucy --path ../datasets
```

> Note that this will preprocess both the ETH and UCY datasets, as they share the same preprocessing script.

</details>


***

### *[UCY](https://github.com/crowdbotpnp/eth_ucy_hl)*: Crowds By Example


<details>
<summary><strong>Dataset details</strong></summary>

<details>
  <summary>Abstract</summary>
  <p style="font-style: italic;">
    Placeholder for the UCY pedestrian dataset abstract text.
  </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
@inproceedings{lerner2007crowds,
  title={Crowds by example},
  author={Lerner, Alon and Chrysanthou, Yiorgos and Lischinski, Dani},
  booktitle={Computer graphics forum},
  volume={26},
  number={3},
  pages={655--664},
  year={2007},
  organization={Wiley Online Library}
}
```

</details>

> #### Dataset Overview
> - The UCY dataset contains pedestrian trajectories recorded in urban environments.
> - Scene types include busy streets and public squares.

</details>

<details>
<summary><strong>Preprocessing instructions</strong></summary>

See instructions for the ETH dataset.

</details>
