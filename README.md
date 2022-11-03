# Algorithm: Gradient descent

Work developed to apply the gradient descent techniques studied during Neural Networks class in the master's degree in Informatics. The work consisted of minimizing the value of the functions using two possible approaches. In the end, some graphics and images in gif format were exported, helping to understand the method. The detailed description of the work is in this repository, in "Trabalho de Redes Neurais.pdf".

## Table of contents
* [How To Use](#how-to-use)
* [Technologies](#technologies)
* [Team](#team)


## How To Use

To clone and run this application, you will need [Github](https://github.com/git-guides/install-git) installed, or you can download the zip file on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/VitordsAmorim/Redes-Neurais.git

# Go into the repository
$ cd Redes-Neurais

# Install dependencies
$ apt install numpy
$ apt install pandas

# Run the code
$ python3 main.py
```


## Technologies
Project is created with:
* Python version: 3.10.6


## Expected outputs

The gradient descent method is applied to each pair of points using the analytically calculated derivative. 

```python
# Given the set of starting points:
initial_points = [[1, 1], [-0.5, -0.5], [0, 0], [0.3, -0.2], [0.7, 1], [1, -0.5]]
```

The solution of the function at each new iteration decreases and converges to minimum points. Thus the goal of finding the parameters which minimize the function was achieved, but there is no guarantee that it is a global minimum. As shown in the image below, the darker the region, the lower the value of the function.

<div align="center">
	<img src="https://github.com/VitordsAmorim/Redes-Neurais/blob/main/Image/my_awesome2.gif" width="480">
</div>





## Team

<div align="left">
  <a href="https://github.com/VitordsAmorim">
  	<img src="https://github.com/VitordsAmorim.png" width="120">
	  
  </a>
	
  <a href="https://github.com/WesleyPereiraPimentel">
  <img src="https://github.com/WesleyPereiraPimentel.png" width="120"/>
  </a>
</div>
	  
  |Vitor Amorim | Wesley Pimentel|
  |:-:|:-:|




	
