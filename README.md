# soft_computing

The project structure follows the answer provided by np8 to [this](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder) Stack Overflow question.

Steps on how the project was structured to handle the imports:

1. Add a setup.py to the root folder (i.e. "soft_computing")
2. Use a virtual environment
    1. Create virtual environment (Linux)<br>
    <code>python -m venv venv</code>
    2. Activate virtual environment (Linux)<br>
    <code>. venv/bin/activate</code>
    3. Deactivate virtual environment (Linux)<br>
    <code>deactivate</code>
3. Pip install your project in an editable state
    1. In the root directory (i.e. "soft_computing"), run:<br>
    <code>pip install -e .</code><br>
    Note the dot, it stands for "current directory". The -e flag installs the project in an editable state. This means that edits to the .py files will be automatically included in the installed package
    2. Verify installation by running:<br>
    <code>pip freeze</code>
4. Import by prepending the "main folder" to every import. An example of this is the "fuzzy" directory that is within "soft_computing".
5. To run the code from a terminal:
    1. Open a terminal and have the current working directory as the root folder of this project (i.e. "soft_computing").
    2. Activate the virtual environment with the above code.
    3. After activating the virtual environment, one may then execute scripts by following this convention:<br>
    <code>python3 ./fuzzy/safin/boston_example.py</code>
6. To run the code from an IDE (e.g. Spyder):
    1. If instead, code is to be run in an IDE such as Spyder, [the Python interpreter will need to be updated](https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment).
    2. Launch Spyder from a terminal where the virtual environment is already activated, and the current working directory is the root of this project (i.e. "soft_computing").
    3. In the taskbar at the top of the window, go to "Tools", and then click on "Preferences". A new window should appear.
    4. In the new window, on the left there should be some options, from top to bottom such as "Appearance", "Run", "Keyboard shortcuts", etc. Click on "Python interpreter".
    5. After clicking on "Python interpreter", to the right of the options should be a section that says: "Select the Python interpreter for all Spyder consoles". Most likely this will be set to "Default (i.e. the same as Spyder's)". This should be changed to "Use the following Python interpreter:".
    6. Once "Use the following Python interpreter:" has been selected, provide/edit the path by clicking on the right button that looks like a page with a folded corner. A new window should appear showing directories and files.
    7. Navigate to the root directory of this project (i.e. "soft_computing"). Click on the "venv" folder; note: this is the folder that was generated when we created the virtual environment, so if it was named differently, act accordingly. However, if you have followed these directions exactly, it should be named "venv".
    8. Inside "soft_computing/venv", click on the "bin" folder. Scroll to the "python3" file, and select it. Now, click "Open". This window will now close.
    9. In the bottom right of the "Preferences" window, click "Apply" and finally click "OK". This window will now close.
