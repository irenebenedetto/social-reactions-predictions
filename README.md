# OSN users reactions analysis

We present a solution for a multi-task regression problem for predicting emotional reactions to posts on Online Social Networks (OSNs). 
Users create posts characterized by textual content and metadata, which can be annotated with nine types of reactions. The prediction model only considers ante-publication information, sets a one-day prediction horizon, and accounts for post creator popularity and specific textual triggers. The study proposes an architecture which combines the XLM-RoBERTa model for text representation with metadata. 

Comparisons are made with baseline models, such as classical regression models and a moving average approach. The Transformer-based regressors leverage XLM-RoBERTa due to its performance and large-scale, multilingual training dataset. To provide explanations of the forecasts, the influence of textual features is studied by analyzing the Transformer component, specifically highlighting the tokens in input text that influence the prediction of a particular reaction type using the SHAP library.

![Model architecture](http://url/to/img.png)
![SHAP](http://url/to/img.png)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package required.

```bash
pip install -r requirements.txt
```

## Usage


02: text only encoded with BERT-based model
04: metadata and text, encoded with BERT

In order to train the baseline models (Machine Learning models) on metadata, insert the data in a specific folder (i.e. `./data`) and run in the terminal:
```bash
python run_baselines_metadata.py --input_folder ./data
```

In order to train the baseline models (Machine Learning models) on textual data, insert the data in a specific folder (i.e. `./data`) and run in the terminal:
```bash
python run_baselines_text.py --input_folder ./data
```

In order to train the baseline models (Machine Learning models) on textual data and metadata, insert the data in a specific folder (i.e. `./data`) and run in the terminal:
```bash
python run_baselines_text_metadata.py --input_folder ./data
```

In order to train XLM-RoBERTa on textual data, insert the data in a specific folder (i.e. `./data`) and run in the terminal:
```bash
python run_xlm_roberta_text.py --input_folder ./data
```

In order to train XLM-RoBERTa on textual data and metadata with the proposed architecture, insert the data in a specific folder (i.e. `./data`) and run in the terminal:
```bash
python run_xlm_roberta_text_metadata.py --input_folder ./data
```



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)