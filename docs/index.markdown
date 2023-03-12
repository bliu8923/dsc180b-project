---
layout: page
title: Home
---

## Introduction

The latest fad in deep learning circles has been the graph neural network (or GNN), a deep learning model
that takes in graph structures and learns on its attributes, whether it be nodes, 
edges, or even entire graphs. However, despite advancements in the field regarding attention and transformers,
many graph networks still struggle to learn over longer distances, defeating the purpose of many
tasks that graph structures encode. Our project aims to benchmark certain networks' performances
on these long range tasks, introduce our own long range benchmark dataset, and use techniques on both the
dataset and network to determine which are best for long range performance.

Our website's main purpose is to visualize our datasets and our results, if you want to see more theoretical portions
of our project (such as the math behind models and optimization techniques), please refer to our report.

## Methodology

For this project, we trained 5 different graph neural networks on 3 datasets and optimized each dataset/model differently to see which
combination would run best. We manipulated data by randomly adding edges throughout the dataset's nodes to decrease average node distances,
and added positional encodings given either by laplacian eigenvectors (LapPE) or by a random walk matrix (RWSE). We also added partial attention
to certain models that supported it.

![](public/encoding-example.png)
<font size="3">
<em>Visualization of the effects of encodings (Courtesy of GraphGPS)</em>
</font>

For each model, we defaulted to using weighted cross entropy as our loss function and Adam as our optimizer, with a learning rate of 0.0001 and
weight decay of 0.9. We also had a scheduler to lower the learning rate when loss plateaued for 15 epochs. Each model was run with 8 layers (except for
SAN, which only had 4 due to memory constraints) and a batch size of 32 (Princeton was 8 due to memory constraints). 

We ran our models on a mix of hardware, either a Nvidia A100 or a Nvidia RTX 3090.

## Datasets

Our models were benchmarked, and performance recorded, on 3 different datasets
with 2 different tasks.

### PascalSP-VOC (Node)

![](public/pascal-example.png)
<font size="3">
<em>Example of Pascal-VOCSP image, and its subsequent conversion to graph form (Courtesy of LRGB)</em>
</font>

Based on the Pascal-VOC 2011 dataset, each image is passed through the SLIC algorithm to create superpixels,
which are each categorized into one of the 21 classes in the original segmentation (20 classes of items, plus 1 for no 
category). The goal of this task is to predict the class of a superpixel region based on 14 node features (12 RGB values,
2-dim point in space representing center-of-mass for superpixel) and 2 edge features (average value of pixels across
boundary, count of pixels on boundary).

Created and sourced from the Long Range Graph Benchmark.

### Peptides-Functional (Graph)
![](public/peptides-example.png)
<font size="3">
<em>Example of 3D peptide structure and it's SMILES graph form (Courtesy of LRGB)</em>
</font>

Based on the SATPdb dataset, 15,535 peptides were converted to graphs through molecular SMILES, with edge and node features
coming from the Open Graph Benchmark feature extraction. Nodes are created from non-hydrogen atoms and edges are created 
from the bonds between these atoms. It should be noted that multi dimensional positional data for these molecules is not
encoded into these graphs whatsoever, and instead graphs prioritize 1D amino acid chains, meaning that the network needs to 
learn its own positions for this dataset.

Created and sourced from the Long Range Graph Benchmark.

### Princeton Shape Benchmark (Graph)

The Princeton Shape Benchmark is a 3D object dataset, in which a model must learn to classify 3D objects based on their shape
into one of many categories. To form these 3D files as graphs, we used a k nearest neighbor method to create edges between the closest *n* neighbors of any
node, and stored its position as features for each node to create a positional encoding.

For this project, we used the "coarse-2" class split, which splits the 3D objects into 7 different classes.
Each object in each class can either be the entire object (i.e. a car, a human) or partial components of something in that 
class (i.e. a wheel, a leaf).

<!-- Import the component -->
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js">

</script>
<style>
    model-viewer {
      height: 300px;
    }
</style>

#### Class 1: Vehicles
<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 135deg" alt="Example 'vehicle' from Princeton Shape Benchmark" src="public/models/m1247.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>
<model-viewer interaction-prompt="none" style="flex-grow: 1; background: white;" id="transform" orientation="0 90deg 225deg" alt="Example 'vehicle2' from Princeton Shape Benchmark" src="public/models/m1551.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>

All types of vehicles made it into this dataset, as well as many common vehicle components. All the vehicle data was very clear,
with good mesh connections and high quality polygons for even the most complex shapes. There was also an extremely diverse pool of data, which included
motorcycles, boats, and helicopters. Components of common vehicles, such as wheels, steering wheels, and more, were also
included in the class. With all of this in mind, the vehicle class was one of the most cleaned classes in the entire dataset and we felt would be 
one of the easier classes to predict, since there are clearer patterns in vehicle shapes.

#### Class 2: Animal

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m259.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m77.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>

The animal class had plenty of variety that included land, sea, and air animals. The sheer number of different animals, including bugs, humans, housepets, 
fish, and birds meant that animals was one of the biggest classes of the dataset. However, unlike the previous vehicles class, the model
could have a hard time predicting on an animal that it did not see before in the dataset, rendering this class much harder to predict compared to a 
common vehicle (such as a car).

Despite these shortcomings, the shapes in the class were well cleaned and intuitive for a human to classify.

#### Class 3: Household

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'household' from Princeton Shape Benchmark" src="public/models/m1729.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'household2' from Princeton Shape Benchmark" src="public/models/m1800.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>

This was one of the most confusing classes of the dataset. The shapes above are intuitively "household" objects, but the object below?


<model-viewer interaction-prompt="none" style="align:left; background: white;" id="transform" orientation="0 90deg 90deg" alt="Example 'household?' from Princeton Shape Benchmark" src="public/models/m722.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>

This sword, along with a fencing sword, a bottle, and even a hat made it into the "household" class. We infer that the "household" class can refer to both components of the actual house,
as well as the household items that can be found in an everyday home, but this creates a much more broad category than the name implies. Almost anything in the household class can be put into another class,
such as the furniture class or the miscellaneous class. Even portions of the household can be classified as "building" items, which would defeat the purpose of seperately classifying building features
and entire buildings.

Besides the misclassifications of this class, most objects seem to be clean and neatly meshed, meaning that a less coarse classification (maybe coarse-1, or base)
would most likely capture the entirety of this class. For the purposes of this project, we kept this class intact.

#### Class 4: Building

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 90deg" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m390.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 45deg" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m460.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
</model-viewer>

The building class is extremely simple and most objects recognizable by humans as everyday buildings. The structure of many resembled typical houses, 
and went so far as futuristic skyscrapers with multiple ladder-like rungs hanging off the walls. In terms of classification, we thought that the class itself
would be a pretty easy one to spot patterns in.

However, some polygons and meshes in this class were messy and ultimately blurred the final product (see the first object displayed). This could have an impact
on the graph created by the model, and ultimately leave our algorithm confused with strange polygons and angles in the mesh.

#### Class 5: Furniture

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 45deg" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m797.glb" camera-controls touch-action="pan-x">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m871.glb" camera-controls touch-action="pan-x">
</model-viewer>

The furniture class includes some more basic household objects and furniture, all of which is well cleaned and easily recognizable.

#### Class 6: Plant

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m1002.glb" camera-controls touch-action="pan-x">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m1080.glb" camera-controls touch-action="pan-x">
</model-viewer>

The plant class goes from household potted plants to full grown trees, all of which have very clean geometry. There were also parts of plants, such as leaves
and stems. In general, the plants are all easily recognizable and their models are cleaned well.

#### Class 7: Miscellaneous

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 45deg" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m575.glb" camera-controls touch-action="pan-x">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 135deg" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m518.glb" camera-controls touch-action="pan-x">
</model-viewer>

This class includes a variety of different models, most of which could go into their own class. Among the models were a lot of faces and silhouettes, as well as multiple
countries modelled in 3D. The models in this class were cleaned well, but the diversity of objects means that this class would be extremely
hard to consistently predict correctly. 

## Results

We used 5 different models, and 3 different optimizations on each of them (mixing and matching).



## Visualizations

## Conclusion

## References

