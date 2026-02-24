# eCAL
Simulator for the eCAL metric


## Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

in case if you want to run LLMs you might need additional dependencies, you can install them by running the following command:
```bash
pip install transformers sentencepiece tiktoken
```
## Usage
To run the calculator, use the following command:
```bash
python RunCalculator.py
```
## Configuration
The configuration is done in the `CalculatorConfig.py` file. What specific configuration options are available can be found in the file itself.
To change the Control and Data plane overheads of the transmission layer or implement new protocols you can change the values  in the `configs/ProtocolConfigs.py` file.

## Citation
If you use this tool please cite our [paper](https://ieeexplore.ieee.org/abstract/document/11298182): 
```
@ARTICLE{11298182,
  author={Chou, Shih-Kai and Hribar, Jernej and Hanžel, Vid and Mohorčič, Mihael and Fortuna, Carolina},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={The Energy Cost of Artificial Intelligence Lifecycle in Communication Networks}, 
  year={2026},
  volume={44},
  number={},
  pages={2427-2443},
  keywords={Artificial intelligence;Measurement;Costs;Energy consumption;Carbon dioxide;Training;Standards;Data centers;Open systems;Energy efficiency;AI model lifecycle;energy consumption;carbon footprint;metric;methodology},
  doi={10.1109/JSAC.2025.3642835}}
```

Other related work:
```
@INPROCEEDINGS{11349371,
  author={Chou, Shih-Kai and Hribar, Jernej and Bertalanič, Blaž and Mohorčič, Mihael and Lagkas, Thomas and Sarigiannidis, Panagiotis and Fortuna, Carolina},
  booktitle={2025 IEEE Conference on Network Function Virtualization and Software-Defined Networking (NFV-SDN)}, 
  title={Energy Cost of the AI/ML Workflow in O-RAN}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Measurement;Adaptation models;Costs;Open RAN;Hardware;Energy efficiency;Complexity theory;Artificial intelligence;Optimization;sustainable 6G networks;O-RAN;AI/ML Workflow;eCAL;lifecycle;energy;Carbon Footprint},
  doi={10.1109/NFV-SDN66355.2025.11349371}}
```

```
@INPROCEEDINGS{10849732,
  author={Chou, Shih-Kai and Hribar, Jernej and Mohorčič, Mihael and Fortuna, Carolina},
  booktitle={2024 IEEE Conference on Standards for Communications and Networking (CSCN)}, 
  title={Towards the Standardization of Energy Efficiency Metrics of the AI Lifecycle in 6G and Beyond}, 
  year={2024},
  volume={},
  number={},
  pages={187-190},
  keywords={Measurement;6G mobile communication;Energy consumption;Costs;Energy measurement;Energy efficiency;Computational efficiency;Quality of experience;Artificial intelligence;Standards;6G;AI-native network;energy efficiency},
  doi={10.1109/CSCN63874.2024.10849732}}
```
