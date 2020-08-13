<link rel="stylesheet" type="./docs/assets/style.css" media="all" href="URL" />

<img align="left" width="auto" height="75" src="./docs/assets/ufmg.png">
<img align="right" width="auto" height="75" src="./docs/assets/verlab.png">
<br/>
<br/>
<br/>
<br/>
<hr>

<h1 align="center"> <b>Learning to Dance - Code </b></h1>

<video align="center" width="auto" controls>
  <source src="./docs/assets/learning_to_dance.mp4" type="video/mp4">
</video>


# Setup:

* There are 3 ways to setup your enviroment to run learning to dance framework. We recommend use an conatiner to avoid issues with system depencies.
## 1 - Installing on your own system:
  You must have installed CUDA and CUDNN with compatible versions with PyTorch 1.4. We use CUDA in version 10.1 and CUDNN in version 7. You can use the cpu version of torch, however you probably have to solve some issues on the requirements.

  ```sudo apt update -y && sudo apt upgrade -y sudo apt install python3-pip ffmpeg -y && sudo pip install -r setup/requirements.txt```
## 2 - Build a Singularity container:
  To build a [Singularity](https://sylabs.io/docs/) conatiner you should have singularity installed in your system at minimum version 2.3. Then you must move the files ```requirements.txt``` and ```singularity``` to the folder we want to build your contanier, then run on the folder you just copy the files:

  ```sudo singularity build NAME_YOUR_CONATINER singularity```

  Also you should take a look in some build flags of singularity (_e.g._ ```--sandbox``` or ```--notest```)

## 3 - Build a Dcoker container:

  To build de image using the Dockerfile run the following command in the ```setup``` folder:

  ```docker build -t NAME_YOUR_IMAGE .```

  After the process of build an image finish, you can acess an shell prompt from the image, creating an container and using the following command:

  ```docker container run -it NAME_YOUR_IMAGE /bin/bash```

# Generate Motion:

* 

# Training:

* 

# bibtex