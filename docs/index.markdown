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
## Datasets

Our models were benchmarked, and performance recorded, on 3 different datasets
with 2 different tasks.

### PascalSP-VOC (Node)

### Peptides-Functional (Graph)

### Princeton Shape Benchmark (Graph)

The Princeton Shape Benchmark is a 3D object dataset, in which a model must learn to classify 3D objects based on their shape
into one of many categories.

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
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 225deg" alt="Example 'vehicle2' from Princeton Shape Benchmark" src="public/models/m1551.glb" camera-controls touch-action="pan-x" shadow-intensity="1">
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

#### Class 6: Plant

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m1002.glb" camera-controls touch-action="pan-x">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 0" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m1080.glb" camera-controls touch-action="pan-x">
</model-viewer>

#### Class 7: Miscellaneous

<model-viewer interaction-prompt="none" style="width: 50%; float: left; background: white;" id="transform" orientation="0 90deg 45deg" alt="Example 'animal' from Princeton Shape Benchmark" src="public/models/m575.glb" camera-controls touch-action="pan-x">
</model-viewer>
<model-viewer interaction-prompt="none" style="margin-left: 50%; background: white;" id="transform" orientation="0 90deg 135deg" alt="Example 'animal2' from Princeton Shape Benchmark" src="public/models/m518.glb" camera-controls touch-action="pan-x">
</model-viewer>

## Results

We used 5 different models, and 3 different optimizations on each of them (mixing and matching).



## Visualizations

## Conclusion