# pscan-classifier
Tool to generate machine learning models to detect port scans

------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Prerequisites

[Open Argus](https://openargus.org/)   3.0.8.2 (argus and clients)

[Nmap](https://nmap.org/)              7.91

[Python](https://www.python.org/)                   3.8.10

[pandas](https://pandas.pydata.org/)               1.2.4

[numpy](https://numpy.org/)                        1.20.3

[Matplotlib](https://matplotlib.org/)              3.4.2

[sklearn](https://scikit-learn.org/)               0.24.2

There are other python dependencies not listed here, but they can be installed on the way.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Usage

This project needs several .argus files, i.e. network flow information files, stored in "./trainData/netflows" folder. These files must have authentical network flows and port scan network flows. You can generate those files using argus and argus clients to record network activity, or converting existing .pcap files to a netflow version (.argus). Refer to [argus documentation](https://openargus.org/using-argus) on how to do that.

One condition to generete these files is to keep track of wich computers in the network are the attackers, and wich ones are innocents, i.e. we need their ips. Then variables.json file needs these ips in scannerIps and targetIps properties respectively. Aditionally it needs the password for sudo privileges when running the trainer.

variables.json
```json
{
	"argusConfig": "./netflowConfFiles",
	"trainingData": "./trainData/netflows",
	"demoData": "./demoData",
	"scannerIps": ["scanner ip here", "scanner ip here"], 
	"targetIps": ["target ip here", "target ip here"] ,
	"password": "password here"
}
```

Finnally running the train.py file will generate a bagging trained model with the following steps:

After dimensional reduction, the correlation matrix of remaining columns is displayed.
<p align="center">
  <img src="https://user-images.githubusercontent.com/44624540/140670179-00e1f2f9-c81c-4cfe-8eea-970232d0c8ad.png?raw=true" alt="Correlation matrix"/>
</p>
At this point the dataframe is ready to be used in training. Once the training ends, two grapichs are displayed, the first decision tree of the bagged model
<p align="center">
  <img src="https://user-images.githubusercontent.com/44624540/140670655-a0a01996-c754-4ae1-aef2-533162843d4a.png?raw=true" alt="Correlation matrix"/>
</p>
And the confusion matrix
<p align="center">
  <img src="https://user-images.githubusercontent.com/44624540/140670714-fdeb1285-7834-4c9d-b46e-85fea215f8f0.png?raw=true" alt="Correlation matrix"/>
</p>
Lastly a column relevance grapich is displayed
<p align="center">
  <img src="https://user-images.githubusercontent.com/44624540/140671570-031e8954-5570-4dd5-a160-ff837d9e1d94.png?raw=true" alt="Correlation matrix"/>
</p>
The model is already created with name bag.pkl.

--------------------------------------------------------------------------------------------------------------------------------

### Demo

To see the model in action use the demo.py file to view a real time netflow clasification. It will search for a model called bag.pkl and it will use argus in daemon mode to fetch the network traffic on the machine.
<p align="center">
  <img src="https://user-images.githubusercontent.com/44624540/140672540-8e0c9af7-acaa-475e-a05e-0b0f680a96c5.png?raw=true" alt="Correlation matrix"/>
</p>
