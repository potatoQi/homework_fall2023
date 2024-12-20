## Install dependencies

There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	其实在 vscode 中使用 conda 并不需要麻烦的在启动菜单打开 Anaconda powershell。直接开一个 vscode 的终端
	然后右下角切换到一个py程序页面，然后右下角把解释器选为cs285的虚拟环境
	然后在终端里conda env list看看是不是cs285虚拟环境，如果是的话，就直接用就行了

	2. Create a conda environment that will contain python 3:
	```
	conda create -n cs285 python=3.9
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	conda activate cs285
	```

	4. Install the requirements into this conda environment
	```
	pip install -r requirements.txt
	```

	5. Allow your code to be able to see 'cs285'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:
	```
	pip install -r requirements.txt
	```

## Troubleshooting 

You may encounter the following GLFW errors if running on machine without a display:

GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
GLFWError: (65537) b'The GLFW library is not initialized'

These can be resolved with:
```
export MUJOCO_GL=egl
```

还可能遇到这个错误：

error: command 'swig.exe' failed: None

需要将swig加入到环境变量中：

先pip show swig，然后会显示swig这个包的安装路径，例如：d:\app\anaconda3\envs\cs285\lib\site-packages

然后进入D:\app\anaconda3\envs\cs285\Scripts\，如果里面有swig.exe，就把D:\app\anaconda3\envs\cs285\Scripts\添加进环境变量的PATH里即可。

还可能遇到这个错误：

ERROR: Failed building wheel for box2d-py
ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (box2d-py)

那么就去https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/，选择 “使用C++的桌面开发” 下载

然后重新运行 pip install -r requirements.txt 即可