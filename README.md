# Traffic_sign_detection_YOLO
Detecting traffic signs using YOLO algorithm
[![IMAGE ALT TEXT HERE](resources/yolo.gif)](https://drive.google.com/file/d/1nxinxEmpOO59KKDkXPgXayNqtq88a_ym/view)


#### Clone the repository
```shell
git clone https://github.com/AmeyaWagh/Traffic_sign_detection_YOLO.git
```

#### Goto darkflow and build cython extension by running
```shell
python3 setup.py build_ext --inplace
```

#### Then build globally with
```shell
pip install .
```

#### Check if "flow" works with "flow --h"
```shell
flow --h
```

#### Go back and create a new folder called "dataset" in base directory. Download and extract LISA dataset into the dataset folder
```shell
cd ..
mkdir dataset
```

#### run datasetGenerator.py
```shell
python3 datasetGenerator.py
```

#### goto darkflow and create "built_graph" directory inside darkflow if you are not training, and save pb and meta files there (pb and meta files can be downloaded here "https://drive.google.com/file/d/171AyNg4zSmz4OXhfcdgU2cxrqTfIV2BD/view?usp=sharing")
```shell
cd darkflow
mkdir built_graph
```

#### set GPU to 0.0 in the config3.json if not using GPU
```json
{
	"yoloConfig":{
		"pbLoad": "./built_graph/tiny-yolo-voc27.pb", 
		"metaLoad": "./built_graph/tiny-yolo-voc27.meta",
		"labels":"../labels.txt",
		"threshold":0.01, 
		"gpu":0.7
	},
	"dataset":"./dataset"	
}
```

#### Run YOLO
```
./runYOLO
``` 

