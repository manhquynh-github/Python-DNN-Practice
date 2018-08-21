# Set up Tensorflow
Most of the projects in this Git will use Tensorflow with Keras as a tool to implement various Deep Neural Networks.
In this project, I will guide you how to set up the environment for Python on Windows.
> [TensorFlow™](https://www.tensorflow.org/) is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices. Originally developed by researchers and engineers from the Google Brain team within Google’s AI organization, it comes with strong support for machine learning and deep learning and the flexible numerical computation core is used across many other scientific domains.

> [Keras](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
## References
- [Install TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
- [github.com/antoniosehk/keras-tensorflow-windows-installation](https://github.com/antoniosehk/keras-tensorflow-windows-installation)
- [github.com/fo40225/tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel)
## Steps
### Step 1. Install Anaconda and Python 3.6
- If you want Microsoft Visual Studio to handle Anaconda and Python components, you can use Visual Studio Installer and install this component:
    - Anaconda3 32/64-bit (5.1.0)
    
- Otherwise, you can just download [Anaconda](https://www.anaconda.com/download/) (Python 3.6 version)
### Step 2. Install Tensorflow
1. __Open Anaconda Prompt__\
Just search in the Start menu. *Note: This prompt is different from normal Command Prompt (cmd)*
2. __(Optional) Create an *environment* for Tensorflow__\
A [conda *environment*](https://conda.io/docs/user-guide/concepts.html#conda-environments) is a directory that contains a specific collection of conda packages that you have installed. If you change one environment, your other environments are not affected. You can easily switch between environments. You can also share your environment with someone by giving them a copy of yours.\
    - Enter the command below to create an *environment* with the name `tensorflow`.\
    `conda create -n tensorflow python=3.6 numpy scipy matplotlib spyder cython`\
    - Enter the command below to activate the *environment*.\
    `activate tensorflow`
3. __Install Tensorflow__
    1. __CPU Version__
    - Enter the command below to install. This will automatically select an official version from Google and install it for you.\
    `pip install --ignore-installed --upgrade tensorflow`
        - If you've installed Tensorflow but failed to import from [Step 3](#step-3-verify-tensorflow-installation), you have to re-compile Tensorflow from the source *__OR__* find a compatible one in this very helpful [Git by fo40225](https://github.com/fo40225/tensorflow-windows-wheel).
            - When you find the correct one, copy the *__link of the file__* and replace `/blob` from the link to `/raw`. The link should end with `.whl` file extenstion and contains no `/blob`.
            - Enter the command below to install. Remember to replace `[the link]` with the modified link.\
                `pip install --ignore-installed --upgrade [the link]`
    2. __GPU Version__
        1. __Check for supported GPU in [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)__\
        Make sure you find your GPU model in the lists and the __Compute Capability__ must be at least 3.0 or else you're out of luck.
        2. __Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)__\
        Current Tensorflow only supports CUDA Toolkit 9.0. For different version, you will need to re-compile Tensorflow from the source. I currently use version 9.2.
        3. __Download and extract [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)__\
        Membership registration is required. Choose the version that suits your OS and CUDA Toolkit version. When extracting the files, remeber the *cuDNN path* to which you're extracting.
        4. __Add the *cuDNN path* to Environment PATH__\
        The path to add is as follow `(extract_path)\cuda\bin`. Tutorials can be found online, you can follow [this one](#add-to-environment-path). To check if PATH is added, close all prompts then open a new Anaconda Prompt and enter the command below and see if the path is in the output result.\
        `echo %PATH%`
        5. Install __Tensorflow with GPU support__.
            - If you installed CUDA Toolkit 9.0, enter the command below\
            `pip install --ignore-installed --upgrade tensorflow-gpu`
            - Otherwise, you have to re-compile Tensorflow from the source *__OR__* follow this [step](#get-windows-tensorflow-wheel by-fo40225).                
4. Wait for the installation to finish.
### Step 3. Verify Tensorflow installation
1. From the current Anaconda Prompt, enter the command below to invoke Python:\
`python`
2. Enter the commands below to verify Tensorflow:
    ```
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    ```
    - The `import tensorflow as tf` is to import Tensorflow. This line can take some time to complete. If everything is correct, nothing will show up and the prompt will wait for another command.
        - If there is a warning __`Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2`__. That means your CPU supports [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) but you've picked the wrong Tensorflow binary of which it was compiled to make use. You should run normally without any problems, but I recommend you use the correct one.
        - If you failed with an error __`ImportError: DLL load failed`__, go back to [Step 2A](#step-2a).4 for more instructions.
    - The `hello = tf.constant('Hello, TensorFlow!')` will create a simple variable using Tensorflow.
    - The `sess = tf.Session()` will create a session upon which Tensorflow will operate. If everything is correct, the prompt will tell you about the info of the CPU/GPU it will use. This will take some time to complete.
    - The `print(sess.run(hello))` will print out the value of `hello` which is supposed to be `b'Hello, TensorFlow!'`.
3. Enter the command below to exit python from shell.\
`exit()`
### Step 4. Install Keras
1. Make sure you are on Anaconda Prompt with *tensorflow environment* activated. Enter the command below to install Keras:\
`pip install keras`
2. Wait for the installation to complete.

## Sub-Steps
### Get [Windows Tensorflow Wheel by fo40225](https://github.com/fo40225/tensorflow-windows-wheel)
1. Find a suitable `.whl` file in the [git](https://github.com/fo40225/tensorflow-windows-wheel).
2. Copy the *__link of the file__* and replace `/blob` from the link to `/raw`. The link should end with `.whl` file extenstion and contains no `/blob`.
3. Open Anaconda Prompt and activate the *environment* if neccessary.
3. Enter the command below to install. Remember to replace `[the link]` with the modified link.\
`pip install --ignore-installed --upgrade [the link]` 
### Add to Environment Path
1. Search for `environment variables` on Start menu.
2. On __System Properties__ popup, click the last button __Environment Variables__.
3. On __Environment Variables__ popup, in __System variables__ combobox, select __Path__ and hit the __Edit__ button.
4. On __Edit environment variables__ popup, hit the __New__ button and copy/paste in *the path*.
5. Click __OK__ for any popups showing up to complete.
*Note: If you hit __Cancel__ then you have to redo the steps*.

*__Please cite this repository if you find this helpful in your work. Thank you.__*
