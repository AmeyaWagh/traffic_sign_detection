# Traffic_sign_detection_YOLO
Detecting traffic signs using YOLO algorithm
[![IMAGE ALT TEXT HERE](resources/frame219.png)](https://drive.google.com/file/d/1nxinxEmpOO59KKDkXPgXayNqtq88a_ym/view)

1) Clone the repo
2) "cd darkflow" and build cython extension by running "python3 setup.py build_ext --inplace"
3) then build globally with "pip install ."
4) check if "flow" works with "flow --h"
5) "cd .." and create a new folder called "dataset". Download and extract LISA dataset into the dataset folder
6) run datasetGenerator.py
7) "cd darkflow" and Create "built_graph" folder inside darkflow if you are not training and save pb and meta files there (pb and meta files can be downloaded here "https://drive.google.com/file/d/171AyNg4zSmz4OXhfcdgU2cxrqTfIV2BD/view?usp=sharing")
8) set GPU to 0.0 in the config3.json if not using GPU
9) RunYOLO 
