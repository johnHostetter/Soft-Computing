# soft_computing

The project structure follows the answer provided by np8 to [this](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder) Stack Overflow question.

Steps on how the project was structured to handle the imports:

1. Add a setup.py to the root folder (i.e. "soft_computing")
2. Use a virtual environment
    1. Create virtual environment INSIDE the "soft_computing" folder (Linux)<br>
    <code>python -m venv venv</code>
    2. Activate virtual environment (Linux)<br>
    <code>. venv/bin/activate</code>
    3. Deactivate virtual environment when you are done (Linux)<br>
    <code>deactivate</code>
3. Pip install your project in an editable state, this lets you access code from the other folders (e.g. code inside "apfrb" can use code from "common")
    1. In the root directory (i.e., "soft_computing"), run:<br>
    <code>pip install -e .</code><br>
    Note the dot, it stands for "current directory". The -e flag installs the project in an editable state. This means that edits to the .py files will be automatically included in the installed package
    2. Verify installation by running:<br>
    <code>pip freeze</code>
4. Read this if you just created a new virtual environment: You will need to pip install all the necessary libraries for the library to work (e.g., "numpy", "sklearn", etc.).
5. Import by prepending the "main folder" to every import. An example of this is the "fuzzy" directory that is within "soft_computing".
6. To run the code from a terminal:
    1. Open a terminal and have the current working directory as the root folder of this project (i.e., "soft_computing").
    2. Activate the virtual environment, "venv", with the above code.
    3. After activating the virtual environment, one may then execute scripts by following this convention:<br>
    <code>python3 ./fuzzy/self_adaptive/boston_example.py</code>
    However, some script files only contain function definitions, so they may not appear to do anything.
7. To run the code from an IDE (e.g., Spyder):
    1. [PyCharm (recommended)] This is by far the simplest and best approach. Download PyCharm Community Edition (for Ubuntu, download via "Ubuntu Software") and launch the program. In the launch screen, you will be prompted if you want to open a project. Click the option for "VCS" ("Version Control Software"), and use this GitHub repository link to clone the project. PyCharm should handle all the necessary setup required then. Although, you may be required to add some missing Python packages (near the bottom of the window).
    2. [Spyder (not recommended)] This approach is more error-prone, and not recommended. Once an error is thrown, the current working directory for which the script files are called is updated, and then requires you to launch the IDE again in order to fix this. Editing the "Current Working Directory" in "Preferences" made no difference too. However, if you insist on this IDE, the following are my old setup instructions, and perhaps you might be able to correct it: If instead, code is to be run in an IDE such as Spyder. First, [a separate virtual environment will need to be made by using Anaconda](https://stackoverflow.com/questions/34217175/spyder-does-not-run-in-anaconda-virtual-environment-on-windows-10). Second, [the Python interpreter will need to be updated](https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment).
    3. To create a virtual environment using Anaconda (note that this is a different virtual environment than what was made above), the following code should be run in a terminal:<br>
    <code>conda create -n env python=3.8 spyder</code><br>
    Note: If you get a warning running the above code, that says a conda environment already exists at "...", then no need to create a new conda environment.
    At the time of writing, Python 3.8.5 is being used, but this will likely be different in the future.
    3. The Anaconda virtual environment is then activated by running:<br>
    <code>conda activate env</code><br>
    Do not use the "venv" virtual environment. Use the "env" conda virtual environment.
    4. Launch Spyder from the terminal where the Anaconda virtual environment is already activated, and the current working directory is the root of this project (i.e., "soft_computing") by simply entering:<br>
    <code>spyder</code><br>
    <strong>The following steps are now obsolete, but are left as a precautionary measure.</strong>
    5. In the taskbar at the top of the window, go to "Tools", and then click on "Preferences". A new window should appear.
    6. In the new window, on the left there should be some options, from top to bottom such as "Appearance", "Run", "Keyboard shortcuts", etc. Click on "Python interpreter".
    7. After clicking on "Python interpreter", to the right of the options should be a section that says: "Select the Python interpreter for all Spyder consoles". Most likely this will be set to "Default (i.e. the same as Spyder's)". This should be changed to "Use the following Python interpreter:".
    8. Once "Use the following Python interpreter:" has been selected, provide/edit the path by clicking on the right button that looks like a page with a folded corner. A new window should appear showing directories and files.
    9. Navigate to the root directory of this project (i.e., "soft_computing"). Click on the "venv" folder; note: this is the folder that was generated when we created the virtual environment, so if it was named differently, act accordingly. However, if you have followed these directions exactly, it should be named "venv".
    10. Inside "soft_computing/venv", click on the "bin" folder. Scroll to the "python3" file, and select it. Now, click "Open". This window will now close.
    11. In the bottom right of the "Preferences" window, click "Apply" and finally click "OK". This window will now close.
