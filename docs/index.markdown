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
combination would run best. We manipulated data by randomly **adding edges** throughout the dataset's nodes to decrease average node distances,
and added positional encodings given either by **laplacian eigenvectors (LapPE)** or by a **random walk matrix (RWSE)**. We also added **partial attention**
to certain models that supported it.

For an extended methodology, refer to our report.

![](public/encoding-example.png)
<font size="3">
<em>Visualization of the effects of encodings (Courtesy of GraphGPS)</em>
</font>
<br>
<font size="3.5">
<table>
  <tbody>
    <tr>
      <th align="center">Datasets</th>
      <th align="center">Model Types</th>
      <th align="center">Optimization Technique</th>
    </tr>
    <tr>
      <td>
        <ul>
          <li>Pascal-VOCSP</li>
          <li>Peptides-func</li>
          <li>Princeton Shape</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Graph Convolutional Network (GCN)</li>
          <li>Residual Gated Graph Convnet (GatedGCN)</li>
          <li>Graph Isomorphic Network (GIN)</li>
          <li>Graph Attention Network (GAT)</li>
          <li>Spectral Attention Network (SAN)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Edge Adding</li>
          <li>Laplacian Positional Encoding (LapPE)</li>
          <li>Random Walk Structural Encoding (RWSE)</li>
          <li>Partial Attention</li>
          <li>Distance Weighted Partial Attention</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
</font>
<font size="3">
<em>Table 1: Datasets we run on, as well as the different model types and techniques we use.</em>
</font>
<br>


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

## Results and Findings

For all datasets, the best model was SAN, with either LapPE or RWSE applied and occasionally edges added.

- Most models, except for PSB, performed best without the added edges. That means that, in order for the models to learn and generalize well on ALL data (not just training data), 
adding edges is unnecessary and even harmful to the overall accuracy of the model. PSB edges were added through k nearest neighbors, so it is possible by doubling them (from 3 to 6)
we were still underfitting the edge count for 3D shapes. 

- SAN trained faster than previously cited (LRGB, GraphGPS), most likely due to advancements in GPU hardware. Training time was brought down from 60 hours in previous benchmarks to 2-3 hours in our benchmark, 
while keeping metrics the same if not higher.

- Laplacian encoding and Random Walk encoding take up time at the beginning of training, since the model needs to calculate
laplacian matrices and eigenvalues before the training begins. This takes up more time the larger the dataset is, and took the longest
on our PSB dataset.

- All models benefitted heavily from a transformer based model, more than just encoding or edge adding could do. PSB saw the biggest improvement just from GCN to SAN.

- Partial with distance weighting seemed to hurt our results in all datasets, and edges/encodings needed to be added for distance weighting to become viable. Distance weighting did not work at all on Pascal.

- Edge adding techniques did not individually improve model performance, but combining edge adding with other encoding techniques or transformers improved model performance compared
to adding encoding or transformers separately.

## Visualizations

[//]: Accuracy, Loss over epoch (selectable for each dataset)
Select a dataset, model, techniques, and metric to visualize!

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v6.js"></script>

<!-- Initialize a select button -->
<select id="selectButton"></select>
<select id="selectButton2"></select>
<select id="selectButton3"></select>
<select id="selectButton4"></select>

<!-- Create a div where the graph will take place -->
<div id="my_dataviz"></div>

<script>

// set the dimensions and margins of the graph
const margin = {top: 30, right: 30, bottom: 30, left: 30},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
    


    //Read the data
d3.json("public/results.json").then( function(data) {

    var defdata = data["GCN"]["pascal"]["base"]
    
    // List of groups (here I have one group per column)
    const allGroup = ["GCN", "GatedGCN", "GIN", "GAT", "SAN"]
    const allData = ["pascal", "peptides", "psb"]
    const allTech = ["base", "edge", "edge+lap", "edge+walk", "walk", "lap"]
    const allMet = ["loss", "train acc", "val acc"]
    
    var rearrangedData = defdata.loss.map(function(d,i) {
      return {
          loss:d,
          epoch:Array.apply(null, Array(500)).map(function (_, i) {return i;})[i]
      }; 
    })
    
    console.log(rearrangedData)
    
    // add the options to the button
    d3.select("#selectButton")
      .selectAll('myOptions')
     	.data(allGroup)
      .enter()
    	.append('option')
      .text(function (d) { return d; }) // text showed in the menu
      .attr("value", function (d) { return d; }) // corresponding value returned by the button

    d3.select("#selectButton2")
      .selectAll('myOptions')
     	.data(allData)
      .enter()
    	.append('option')
      .text(function (d) { return d; }) // text showed in the menu
      .attr("value", function (d) { return d; }) // corresponding value returned by the button

    d3.select("#selectButton3")
          .selectAll('myOptions')
            .data(allTech)
          .enter()
            .append('option')
          .text(function (d) { return d; }) // text showed in the menu
          .attr("value", function (d) { return d; }) // corresponding value returned by the button

    d3.select("#selectButton4")
          .selectAll('myOptions')
            .data(allMet)
          .enter()
            .append('option')
          .text(function (d) { return d; }) // text showed in the menu
          .attr("value", function (d) { return d; }) // corresponding value returned by the button

    // A color scale: one color for each group
    const myColor = d3.scaleOrdinal()
      .domain(allGroup)
      .range(d3.schemeSet2);

    // Add X axis --> it is a date format
    const x = d3.scaleLinear()
      .domain([0,500])
      .range([ 0, width ]);
    svg.append("g")
      .attr("transform", `translate(0, ${height})`)
      .call(d3.axisBottom(x));

    // Add Y axis
    const y = d3.scaleLinear()
      .domain( [Math.min(...defdata.loss)-0.01,Math.max(...defdata.loss) + 0.1])
      .range([ height, 0 ]);
    
    svg.append("g")
      .attr("id", "y-axis")
      .call(d3.axisLeft(y));
      
    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function(d) { return d.epoch; }).left;
    
    // Create the circle that travels along the curve of chart
    var focus = svg
    .append('g')
    .append('circle')
      .style("fill", "none")
      .attr("stroke", "black")
      .attr('r', 8.5)
      .style("opacity", 0)
  

    // Initialize line with group a
    const line = svg
      .append('g')
      .append("path")
        .datum(rearrangedData)
        .attr("d", d3.line()
          .x(function(d) { return x(+d.epoch) })
          .y(function(d) { return y(+d.loss) })
        )
        .attr("stroke", "blue")
        .style("stroke-width", 4)
        .style("fill", "none")
    
    // Create the text that travels along the curve of chart
    var focusText = svg
    .append('g')
    .append('text')
      .style("opacity", 0)
      .attr("text-anchor", "left")
      .attr("alignment-baseline", "middle")
      .attr("font", "SF Pro Text")
      .attr("color", "gray")
        
    // Create a rect on top of the svg area: this rectangle recovers mouse position
    svg
    .append('rect')
    .style("fill", "none")
    .style("pointer-events", "all")
    .attr('width', width)
    .attr('height', height)
    .on('mouseover', mouseover)
    .on('mousemove', mousemove)
    .on('mouseout', mouseout);
    
    // What happens when the mouse move -> show the annotations at the right positions.
    function mouseover() {
        focus.style("opacity", 1)
        focusText.style("opacity",1)
    }
    
    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.pointer(event)[0]);
        var i = bisect(rearrangedData, x0, 1);
        selectedData = rearrangedData[i]
        focus
          .attr("cx", x(selectedData.epoch))
          .attr("cy", y(selectedData.loss))
        if (i > 250){
            focusText
              .html("epoch:" + selectedData.epoch + "  -  " + "loss:" + Math.round((selectedData.loss + Number.EPSILON) * 1000) / 1000)
              .attr("x", x(selectedData.epoch)-225)
              .attr("y", y(selectedData.loss)-50)
        } else {
            focusText
              .html("epoch:" + selectedData.epoch + "  -  " + "loss:" + Math.round((selectedData.loss + Number.EPSILON) * 1000) / 1000)
              .attr("x", x(selectedData.epoch)+10)
              .attr("y", y(selectedData.loss)-50)
        }
    }
    function mouseout() {
        focus.style("opacity", 0)
        focusText.style("opacity", 0)
    }
    
    // A function that update the chart
    function update(g1, g2, g3, g4) {
    
      if (g4 == "train acc"){
          g4 = "tacc";
      }
      if (g4 == "val acc"){
          g4 = "vacc";
      }
    
      var defdata = data[g1][g2][g3]

      // Create new data with the selection?
      var rearrangedData = defdata[g4].map(function(d,i) {
          return {
              loss:d,
              epoch:Array.apply(null, Array(500)).map(function (_, i) {return i;})[i]
          }; 
        })
        
      // Add Y axis
        const y = d3.scaleLinear()
          .domain( [Math.min(...defdata[g4])-0.01,Math.max(...defdata[g4]) + 0.1])
          .range([ height, 0 ]);
          
        svg.select("#y-axis")
            .transition()
            .call(d3.axisLeft(y));
     
      // Give these new data to update line
        line
            .datum(rearrangedData)
            .transition()
            .duration(1000)
            .attr("d", d3.line()
            .x(function(d) { return x(+d.epoch) })
            .y(function(d) { return y(+d.loss) })
            );
    svg
    .select('rect')
    .style("fill", "none")
    .style("pointer-events", "all")
    .attr('width', width)
    .attr('height', height)
    .on('mouseover', mouseover)
    .on('mousemove', mousemove)
    .on('mouseout', mouseout);
         
        
       // What happens when the mouse move -> show the annotations at the right positions.
    function mouseover() {
        focus.style("opacity", 1)
        focusText.style("opacity",1)
    }
    
    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.pointer(event)[0]);
        var i = bisect(rearrangedData, x0, 1);
        selectedData = rearrangedData[i]
        focus
          .attr("cx", x(selectedData.epoch))
          .attr("cy", y(selectedData.loss))
        if (i > 250){
            focusText
              .html("epoch:" + selectedData.epoch + "  -  " + g4 + " :" + Math.round((selectedData.loss + Number.EPSILON) * 1000) / 1000)
              .attr("x", x(selectedData.epoch)-225)
              .attr("y", y(selectedData.loss)-50)
        } else {
            focusText
              .html("epoch:" + selectedData.epoch + "  -  " + g4 + " :" + Math.round((selectedData.loss + Number.EPSILON) * 1000) / 1000)
              .attr("x", x(selectedData.epoch)+10)
              .attr("y", y(selectedData.loss)-50)
        }
    }
    function mouseout() {
        focus.style("opacity", 0)
        focusText.style("opacity", 0)
    }
    }

    // When the button is changed, run the updateChart function
    d3.select("#selectButton").on("change", function(event,d) {
        // recover the option that has been chosen
        const g1 = d3.select("#selectButton").property("value")
        const g2 = d3.select("#selectButton2").property("value")
        const g3 = d3.select("#selectButton3").property("value")
        const g4 = d3.select("#selectButton4").property("value")
        // run the updateChart function with this selected option
        update(g1, g2, g3, g4)
    })
    d3.select("#selectButton2").on("change", function(event,d) {
        // recover the option that has been chosen
        const g1 = d3.select("#selectButton").property("value")
        const g2 = d3.select("#selectButton2").property("value")
        const g3 = d3.select("#selectButton3").property("value")
        const g4 = d3.select("#selectButton4").property("value")
        // run the updateChart function with this selected option
        update(g1, g2, g3, g4)
    })
    d3.select("#selectButton3").on("change", function(event,d) {
        // recover the option that has been chosen
        const g1 = d3.select("#selectButton").property("value")
        const g2 = d3.select("#selectButton2").property("value")
        const g3 = d3.select("#selectButton3").property("value")
        const g4 = d3.select("#selectButton4").property("value")
        // run the updateChart function with this selected option
        update(g1, g2, g3, g4)
    })
    d3.select("#selectButton4").on("change", function(event,d) {
        // recover the option that has been chosen
        const g1 = d3.select("#selectButton").property("value")
        const g2 = d3.select("#selectButton2").property("value")
        const g3 = d3.select("#selectButton3").property("value")
        const g4 = d3.select("#selectButton4").property("value")
        // run the updateChart function with this selected option
        update(g1, g2, g3, g4)
    })

})
</script>


[//]: Scatter plot performance over time (1 per dataset)



## References
Dwivedi, Vijay Prakash and Rampášek, Ladislav and Galkin, Mikhail and Parviz, Ali and Wolf, Guy and Luu, Anh Tuan and Beaini, Dominique. [*Long Range Graph Benchmark*.](https://arxiv.org/abs/2206.08164) arXiv:2206.08164. Jan 16, 2023.

Galkin, Michael. [*GraphGPS: Navigating Graph Transformers*.](https://urldefense.com/v3/__https://na.eventscloud.com/urc2023__;!!Mih3wA!DtYED_AvtXIaJjcoBbyq5sdp7pD0aFGcOJ5CffiMU2Beq0gBdAxJ5Taj0DlsE_uxns1Qzyw38MX9yZ8vub-7$) June 13, 2022. 

Shilane, Philip and Min, Patrick and Kazhdan, Michael and Funkhouser, Thomas. [*Princeton Shape Benchmark*.](https://shape.cs.princeton.edu/benchmark/benchmark.pdf) Shape Modeling International. June, 2004.