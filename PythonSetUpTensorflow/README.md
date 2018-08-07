# Set up Tensorflow
Most of the projects in this Git will use Tensorflow with Keras as a tool to implement various Deep Neural Networks.
In this project, I will guide you how to set up the environment for Python on Windows.
> [TensorFlow™](https://www.tensorflow.org/) is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices. Originally developed by researchers and engineers from the Google Brain team within Google’s AI organization, it comes with strong support for machine learning and deep learning and the flexible numerical computation core is used across many other scientific domains.

> [Keras](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
## Reference
- [github.com/antoniosehk/keras-tensorflow-windows-installation](https://github.com/antoniosehk/keras-tensorflow-windows-installation)
- [github.com/fo40225/tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel)
## Steps
### Step 1. Install Anaconda and Python 3.6
a. If you want Microsoft Visual Studio to handle Anaconda and Python components, you can use Visual Studio Installer and install these components:
- Anaconda3 32/64-bit (5.1.0)
- Python3 32/64-bit (3.6.5)

b. Otherwise, you can just download [Anaconda](https://www.anaconda.com/download/) (Python 3.6 version)
### Step 2. Install Tensorflow
For those machines that have CPU only, go to __(Step 2A)__, those machines that have NVIDIA GPU go to __(Step 2B)__
#### Step 2A
1. Open Anaconda Prompt (just search in the Menu Start). *Note: This promt is different from normal Command Prompt (cmd)*
2. Enter the command below to create an *environment* with the name `tensorflow` or your own choice.\
`conda create -n tensorflow python=3.6 numpy scipy matplotlib spyder`
3. Enter the command below to activate the *environment*.\
`activate tensorflow`
4. Enter the command below to install Tensorflow with __CPU__ support and wait for the installation to finish then jump to Step 3.\
`pip install --ignore-installed --upgrade tensorflow`
#### Step 2B
1. Check for compatible GPU support in [CUDA NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus). Make sure you find your GPU model in the lists and the Compute Capability must be at least 3.0 or else you're out of luck.
2. Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Current Tensorflow only supports CUDA Toolkit 9.0. For higher CUDA Toolkit version, you will need to re-compile Tensorflow from the source. I currently use version 9.2.
3. Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-download). Membership registration is required. Choose the version depending on  your operating system and CUDA Toolkit version. When extracting the files, remeber *the cuDNN path* to which you're extracting.
4. Add *the said cuDNN path from previous step* into Environment PATH. Multiple tutorials of how to add to Environment PATH can be found online, you can follow this one. The path is as follow\
`(extract_path)\cuda\bin`
    - Search for `environment variables` on Menu Start.
    - On __System Properties__ popup, click the last button __Environment Variables__.
    - On __Environment Variables__ popup, in __System variables__ combobox, select __Path__ and hit the __Edit__ button.
    - On __Edit environment variables__ popup, hit the __New__ button and copy/paste in *the said path*.
    - Click __OK__ for any popups showing up to complete. *Note: If you hit __Cancel__ then you have to redo the steps*.
5. Check if PATH is added. Close all prompts then open a new Anaconda Prompt, enter the command below and see if the path is in the output result.\
`echo %PATH%`
6. Enter the command below to create an *environment* with the name `tensorflow` or your own choice.\
`conda create -n tensorflow python=3.6 numpy scipy matplotlib spyder`
7. Enter the command below to activate the *environment*.\
`activate tensorflow`
8. Install Tensorflow
    - If you installed CUDA Toolkit 9.0, enter the command below\
    `pip install --ignore-installed --upgrade tensorflow-gpu`
    - Otherwise, you have to build the Tensorflow binaries on your own *___OR__* find compatible one in this very helpful [Git by fo40225](https://github.com/fo40225/tensorflow-windows-wheel).
        - When you find the correct one, copy the *__link of the file__* and replace `/blob` from the link to `/raw`. The link should end with `.whl` file extenstion and contains no `/blob`.
        - Enter the command below to install Tensorflow with __GPU__ support.\
        `pip install --ignore-installed --upgrade [the link]` 
    - Wait for the installation to finish then jump to Step 3.
### Step 3. Verify Tensorflow installation
1. From the current Anaconda Prompt enter the commands below:\
`python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))`
- The `python` line is to invoke python from shell.
- The `import tensorflow as tf` is to import tensorflow. This line can take some time to complete. If everything is correct, nothing will show up and the prompt will wait for another command. If there is a warning `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2`. That means your CPU supports [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) but you've picked the wrong Tensorflow binary which was compiled to make use of. You should run normally without any problems, but I recommend you use the correct one.
- The `hello = tf.constant('Hello, TensorFlow!')` will create a simple variable using Tensorflow.
- The `sess = tf.Session()` will create a session upon which Tensorflow will operate. If everything is correct, the prompt will tell you about the info of the CPU/GPU it will use. This will take some time to complete.
- The `print(sess.run(hello))` will print out the value of `hello` which is supposed to be `b'Hello, TensorFlow!'`.
