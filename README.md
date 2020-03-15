# AMTrafficPhase2-Task_#
Detection of faces and licene plates with different methods

Data to check speed/ visually assess performance:
-EU number plates https://drive.google.com/open?id=1aI0Ug1SqQgOZKWTDH8xr7996wUNCsW0I
-Faces [to be added]

Methods:
-LP: LBP cascade, []
-Face: [to be added]


## HOW TO TEST

0) run ```pip3 install -r requirements.txt```

### License plates
- Cascade:
1) Download sample frames and .xml 
2) Put the downloaded files in root folder
3) run ```python3 test_casade.py --in_folder {./eu} --out_path {path to save samples} --cascade_path {.xml file}```


The script will save images with license plates highlighted, and print inference time to terminal.