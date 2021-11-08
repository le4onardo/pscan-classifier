# pscan-classifier
Tool to generate machine learning models to detect port scans


### Prerequisites

[Open Argus](https://openargus.org/)   3.0.8.2 (argus and clients)

[Nmap](https://nmap.org/)              7.91

[Python](https://www.python.org/)                   3.8.10

[pandas](https://pandas.pydata.org/)               1.2.4

[numpy](https://numpy.org/)                        1.20.3

[Matplotlib](https://matplotlib.org/)              3.4.2

[sklearn](https://scikit-learn.org/)               0.24.2

There are other python dependencies not listed here, but they can be installed on the way.

### Usage

This project needs several .argus files, i.e. network flow information files, stored in "./trainData/netflows" folder. These files must have authentical network flows and port scan network flows. You can generate those files using argus and argus clients to record network activity, or converting existing .pcap files to a netflow version (.argus). Refer to [argus documentation](https://openargus.org/using-argus) on how to do that.

One condition to generete these files is to keep track of wich computers in the network are the attackers, and wich ones are innocents, i.e. we need their ips. Then variables.json file needs these ips in scannerIps and targetIps properties respectively. Aditionally it needs the password for sudo privileges when running the trainer.

variables.json
```json
{
	"argusConfig": "./netflowConfFiles",
	"trainingData": "./trainData/netflows",
	"demoData": "./demoData",
	"scannerIps": ["replace here", "replace here"], 
	"targetIps": ["replace here", "replace here"] ,
	"password": "replace here"
}
```


Finnally running the train.py file will generate a bagging trained model with the following steps:

After dimensional reduction, the correlation matrix of remaining columns is displayed:

[correlation matrix](https://user-images.githubusercontent.com/44624540/140670179-00e1f2f9-c81c-4cfe-8eea-970232d0c8ad.png)


