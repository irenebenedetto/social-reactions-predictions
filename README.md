# OSN users reactions analysis
We present a solution for a multi-task regression problem for predicting emotional reactions to posts on Online Social Networks (OSNs). 
Users create posts characterized by textual content and metadata, which can be annotated with nine types of reactions. The prediction model only considers ante-publication information, sets a one-day prediction horizon, and accounts for post creator popularity and specific textual triggers. The study proposes an architecture which combines the XLM-RoBERTa model for text representation with metadata. 

Comparisons are made with baseline models, such as classical regression models and a moving average approach. The Transformer-based regressors leverage XLM-RoBERTa due to its performance and large-scale, multilingual training dataset. To provide explanations of the forecasts, the influence of textual features is studied by analyzing the Transformer component, specifically highlighting the tokens in input text that influence the prediction of a particular reaction type using the SHAP library.

