# TSUNAMI-FORECASTING-USING-MULTIVARIATE-LSTM-RNN

FINAL PROJECT PROPOSAL


TSUNAMI FORECASTING USING MULTIVARIATE LONG SHORT-TERM MEMORY (LSTM) RECURRENT NEURAL NETS (RNNs)


 




IMRAN Y. A. ABU LIBDA

D121211105






BACHELOR DEGREE PROGRAM
INFORMATICS ENGINEERING DEPARTMENT 
FACULTY OF ENGINEERING
HASANUDDIN UNIVERSITY
MAKASSAR
2024
 
I.	Title 
Tsunami Forecasting Using Multivariate Long Short-Term Memory (LSTM) Recurrent Neural Nets (RNNs)
II.	Background
Tsunami forecasting is a critical task for authorities, emergency responders, and communities in coastal areas, as it can provide vital insights into the likelihood and severity of tsunami events. Accurate tsunami forecasts can aid in evacuating people, protecting infrastructure, and minimizing damage. However, predicting tsunamis is challenging due to the complex and dynamic nature of oceanographic and seismic systems, and the various factors that can influence tsunami generation and propagation.
Traditionally, tsunami forecasting has relied on physical models and seismic data analysis to understand the relationships between tsunami waves and oceanographic and seismic variables, such as sea surface displacement, ocean depth, and earthquake magnitude (Bashiri, 2021). While these methods can provide valuable insights, they can also be limited in their ability to capture the complexity and dynamics of tsunami events.
In recent years, machine learning techniques have shown promising results in the field of time series forecasting, particularly in the use of recurrent neural networks (RNNs) such as Long Short-Term Memory (LSTM) networks. These networks are well-suited for sequential data analysis, such as time series forecasting, and have been shown to outperform traditional statistical methods in various applications (Roser, 2021).
This project aims to develop a multivariate LSTM RNN model that can forecast tsunami events by analyzing various oceanographic and seismic parameters. By leveraging the strengths of machine learning, this approach can potentially improve the accuracy and reliability of tsunami forecasting, enabling more effective early warning systems and emergency preparedness.
The significance of this research lies in its potential to save lives and reduce damage to infrastructure in coastal areas. Accurate tsunami forecasting can provide critical minutes or even hours for evacuation, allowing communities to seek safety and avoid the devastating impacts of tsunamis. Moreover, this research can contribute to the development of more effective early warning systems, enhancing the resilience of coastal communities worldwide.
This project proposes to use machine learning techniques, specifically a combination of RNN and LSTM with multivariate features, to forecast tsunami events. By using both RNN and LSTM, we aim to capture both short-term and long-term dependencies in the data, leading to more accurate predictions of tsunami events.
In recent years, machine learning techniques have shown promising results in the field of time series forecasting, particularly in the use of recurrent neural networks (RNNs) such as Long Short-Term Memory (LSTM) networks. These networks are well-suited for sequential data analysis, such as time series forecasting, and have been shown to outperform traditional statistical methods in various applications (Roser, 2021).
In this project, I propose to use machine learning techniques, specifically a combination of RNN and LSTM with multivariate features, to forecast inflation rates in Indonesia. The LSTM is a type of RNN that is well-suited for tasks that involve sequential data, such as time series forecasting, and has been shown to achieve good performance in a range of applications. By using both RNN and LSTM, I aim to capture both short-term and long-term dependencies in the data, leading to more accurate predictions of inflation (Lee, 2020).
To accomplish this, I have collected a comprehensive dataset spanning 16 years of various oceanographic and meteorological variables from sources such as Copernicus Climate Data Store (ERA5 Hourly Data on Single Levels) and BMKG (Badan Meteorologi, Klimatologi, Dan Geofisika). The dataset contains 4 types of data variables that are known to have an impact on tsunami formation, which is Seismic Data, Oceanographic Data, Geological Data, and Meteorological Data.
One of the main challenges of working with time series data is the issue of temporal dependencies. Oceanographic and meteorological variables can influence tsunami events in complex ways, and their effects can change over time. This is where the combination of RNN and LSTM can be particularly effective. RNNs can capture the temporal dependencies in the data and extract important features, which can then be fed into the LSTM for more accurate prediction. In this research, I plan to use an RNN for the initial processing of the data and then use an LSTM for the final prediction (Bontempi, 2019).
The dataset will be preprocessed and cleaned to remove any missing or erroneous data, and the relevant features will be selected based on their impact on tsunami events. The model will be trained using a combination of supervised and unsupervised learning techniques, and performance metrics such as mean absolute error (MAE) and root mean square error (RMSE) will be used to evaluate the accuracy of the model (James, 2013).
Once the model is trained, it will be available online through a web application that will be developed using Plotly, a popular open-source data visualization library that provides a variety of interactive charting tools and graphs. The app will be designed with user-friendliness in mind and will provide users with easy access to the most important information. It will be used to make predictions about future tsunami events and provide early warnings. Overall, the web app will serve as a useful tool for disaster management agencies, coastal communities, and other stakeholders to access the tsunami forecasting model and make informed decisions based on the predicted tsunami events.

Overall, this research aims to develop a reliable and accurate method for forecasting tsunamis using multivariate features and LSTM RNNs. The results of this research can have significant implications for disaster management, coastal community planning, and government policies in regions prone to tsunami threats.

Keywords: Tsunami, Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), Forecasting.






III.	Problem Statement
Based on the background that has been described, the formulation of the problem to be solved in this study is:
1.	How can multivariate oceanographic and meteorological variables be used with the Long Short-Term Memory (LSTM) algorithm to accurately predict tsunami occurrences?

2.	How does the combination of LSTM and RNN impact the accuracy of tsunami predictions compared to using only one of the algorithms?

3.	How can the model be validated and applied to real-world tsunami warning systems to support decision-making for emergency responses and coastal management?

4.	How can the tsunami forecasting model be integrated into an online platform to provide accessible and real-time predictions for stakeholders?


IV.	Research Objectives
The aim of this research is:
1.	Develop a model that accurately predicts tsunami occurrences using a combination of LSTM RNNs and multivariate oceanographic and meteorological variables.
2.	Evaluate the performance of the LSTM RNN model compared to other forecasting methods, such as traditional statistical or machine learning models.
3.	Identify the key oceanographic and meteorological variables that are most closely associated with tsunami occurrences to evaluate their relative importance in predicting future events.
4.	Evaluate the effectiveness of the system in predicting tsunami occurrences and compare its performance to other forecasting methods.

5.	Implement a system that utilizes real-time oceanographic and meteorological data to make tsunami forecasts using the LSTM RNNs model.

V.	Research Benefits
The benefits of this research are:
1.	For policymakers can assist in making informed decisions about tsunami preparedness and response strategies, potentially saving lives and reducing economic losses. And provide valuable insights into the impact of oceanographic and meteorological conditions on tsunami formation.
2.	For emergency management agencies can enhance the ability to predict and respond to tsunami threats in a timely manner, improving the efficiency and effectiveness of evacuation plans and other emergency response actions. And contribute to the development of early warning systems that can provide real-time alerts to vulnerable coastal communities
3.	For coastal communities can empower residents with timely and accurate information about potential tsunami threats, enabling them to take necessary precautions and evacuate if needed. Also, to increase public awareness and understanding of tsunami risks and preparedness measures, fostering a culture of safety and resilience.
4.	For businesses and infrastructure planners to provide critical information for risk assessment and mitigation planning, helping businesses and infrastructure planners to design and implement measures to protect assets and ensure continuity during tsunami events. Also, aid in the development of insurance models that accurately reflect the risk of tsunami damage, potentially reducing financial losses for businesses and property owners.
5.	For academic researchers to contribute to the existing knowledge base on tsunami forecasting and the application of machine learning techniques in natural disaster prediction. And provide insights and methodologies that can inform future research in the fields of oceanography, meteorology, and disaster risk management.

VI.	Research Limitations
Limitations of the problem of this study are:
1.	Geographic scope: The study targets tsunami forecasting for Indonesia only, which included in the sub-region: N.17°, W.84°, E.135°, S.5°.
2.	Data limitations: The used data is from 2008 until 2023 on an hourly frequency.
3.	Forecast horizon: The short-term nature of the forecast horizon may limit the usefulness of the findings for longer-term planning and decision-making.
4.	Validation and generalization: The validation of the model may be limited by the availability of historical tsunami events in the dataset.






VII.	Related research

NO	Title	Author	Publisher	Method	Results
1.	Tsunami tide prediction in shallow water using recurrent neural networks: model implementation in the Indonesia Tsunami Early Warning System	M. A. Nurdin, D. D. Susanto, S. D. Prayogo, A. Muchtar, and M. I. Syamsuddin	Springer	Long Short-Term Memory (LSTM), recurrent neural network (RNN)	The GRU model predicted tides with high accuracy and identified potential tsunamis by analyzing z-scores, making it a valuable tool for mitigating tsunami damage.
2.	Machine Learning Algorithms for Real-time Tsunami Inundation Forecasting: A Case Study	Machine Learning Algorithms for Real-time Tsunami Inundation Forecasting: A Case Study	Pure and Applied Geophysics	Convolutional neural network (CNN) and a multilayer perceptron (MLP)	The results show that the proposed methods are extremely fast (less than 1 s) and comparable with nonlinear forward modeling.
3.	Tsunami Forecasting Using Artificial Neural Networks	M. A. Javed	IEEE	Artificial Neural Network (ANN)	The review demonstrated that ANN methods can provide accurate and rapid tsunami forecasting.
4.	Artificial neural network for tsunami forecasting	Michele Romano, Shie-Yui Liong, Minh Tue Vu, Pavlo Zemskyy, Chi Dung Doan, My Ha Dao, Pavel Tkalich	Elsevier Ltd	Artificial Neural Network (ANN)	The validation tests demonstrated that with a given earthquake size and location, the ANN method provides accurate and near instantaneous forecasting of the maximum tsunami heights and arrival times for the entire computational domain.
5.	Earthquake trend prediction using long short-term memory RNN	Vardaan, K., Tanvi Bhandarkar, Nikhil Satish, S. Sridhar, R. Sivakumar, and Snehasish Ghosh	International Journal of Electrical and Computer Engineering

	Long Short-Term Memory (LSTM)	The LSTM neural network outperformed the ordinary Feed Forward Neural Network (FFNN) and the R^2 score of the LSTM is better than the FFNN’s by 59%.
6.	The Predictability of the 30 October 2020 İzmir-Samos Tsunami Hydrodynamics and Enhancement of Its Early Warning Time by LSTM Deep Learning Network	Alan, Ali Rıza, Cihan Bayındır, Fatih Ozaydin, and Azmi Ali Altintas	Water	LSTM Recurrent Neural Network	The LSTM network effectively predicted tsunami hydrodynamics and enhanced early warning times, outperforming traditional methods, with computation times around 52 s for the data sets analyzed.
7.	Accurate tsunami wave prediction using long short-term memory based neural networks	Xu, Hang, and Huan Wu	Ocean Modelling	Long short-term memory (LSTM) based neural networks	The results show that the proposed LSTM-based neural networks outperformed traditional methods in terms of accuracy and computational efficiency.





VIII.	System Proposal

Objective
This section outlines the proposed system for tsunami forecasting using a multivariate LSTM RNN model. The primary goal is to leverage machine learning techniques to predict tsunami events more accurately by analyzing oceanographic and seismic data.
 
 

Dataset
The dataset consists of 16 years of oceanographic and seismic data obtained from the Copernicus Climate Data Store (ERA5) and the Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). The combined dataset includes the following variables:

ERA5 Data:
- Longitude (lon)
- Latitude (lat)
- Timestamp (time)
- U-component of Wind (u10)
- V-component of Wind (v10)
- Mean Sea Level Pressure (msl)
- Sea Surface Temperature (sst)
- Significant Wave Height (swh)
- Surface Pressure (sp)
- Temperature at 2 Meters (t2m)
- Total Precipitation (tp)

BMKG Data:
- Event Identifier (eventID)
- Timestamp (datetime)
- Latitude (latitude)
- Longitude (longitude)
- Magnitude (magnitude)
- Magnitude Type (mag_type)
- Depth (depth)
- Number of Phases (phasecount)
- Azimuthal Gap (azimuth_gap)

Data Processing and Insights

Correlation Matrix Analysis:
The correlation matrix for the combined dataset reveals several key relationships between the variables, which are critical for developing the forecasting model. The primary insights include:

1. Longitude and Latitude:
   - Longitude is negatively correlated with the V-component of wind (v10) and Significant Wave Height (swh), indicating lower values for these variables at higher longitudes.
   - Latitude has a moderate positive correlation with Sea Surface Temperature (sst), suggesting higher sea surface temperatures at higher latitudes.

2. Wind Speed at 10 Meters (u10 and v10):
   - Eastward wind component (u10) is positively correlated with Significant Wave Height (swh), indicating stronger eastward winds lead to higher wave heights.
   - Northward wind component (v10) is correlated with both u10 and swh, suggesting a relationship between wind components and wave heights.

3. Mean Sea Level Pressure (msl):
   - MSL is positively correlated with Surface Pressure (sp) and Temperature at 2 Meters (t2m), indicating that higher pressures are associated with higher air temperatures.

4. Sea Surface Temperature (sst):
   - SST has strong positive correlations with t2m and latitude, implying that higher sea surface temperatures are associated with higher air temperatures and latitudes.

5. Significant Wave Height (swh):
   - SWH is correlated with wind components (u10 and v10) and negatively correlated with SST and MSL, indicating higher wave heights are related to stronger winds and lower sea surface temperatures and pressures.

6. Surface Pressure (sp):
   - SP is perfectly correlated with MSL and has strong positive correlations with t2m and SST, indicating a strong relationship between surface pressure and temperatures.

7. Air Temperature at 2 Meters (t2m):
   - T2M is positively correlated with SST and SP, confirming the relationship between air temperature, sea surface temperature, and surface pressure.

8. Total Precipitation (tp):
   - TP shows weak correlations with most variables, indicating that precipitation is not strongly related to other variables in this dataset.

9. Magnitude and Depth of Earthquakes:
   - Magnitude has moderate correlations with phasecount and depth, suggesting larger earthquakes tend to have more phases recorded and occur at greater depths.

10. Phasecount and Azimuth Gap:
    - Phasecount shows positive correlations with magnitude and depth, implying that earthquakes with higher magnitudes and depths have more recorded phases.
    - Azimuth gap has negative correlations with magnitude and phasecount, suggesting that larger earthquakes and those with more phases recorded tend to have smaller azimuth gaps.

 Informative Analysis and Insights
The insights gained from the correlation matrix analysis will guide the feature selection and model development processes. The key relationships identified will help in understanding the underlying patterns in the data, leading to more accurate predictions.

1. Temperature Relationships: The strong relationship between air temperature (t2m), sea surface temperature (sst), and surface pressure (sp) highlights the importance of these variables in the model.
2. Wind and Waves: The close relationship between wind speed components and significant wave height (swh) indicates that wind data is crucial for predicting wave heights, which can impact tsunami generation.
3. Earthquake Characteristics: Understanding the correlations between earthquake magnitude, depth, and phasecount is essential for incorporating seismic data into the model.
4. Spatial Variations: The varying correlations between longitude and latitude with different variables highlight the importance of spatial analysis in tsunami forecasting.

Proposed System
The proposed system will integrate the insights from the data analysis into a multivariate LSTM RNN model for tsunami forecasting. The system will include the following components:

1. Data Preprocessing: Cleaning and preparing the data, including handling missing values and normalizing the data.
2. Feature Selection: Selecting the most relevant features based on the correlation analysis to improve model performance.
3. Model Development: Developing and training the multivariate LSTM RNN model using the processed dataset.
4. Evaluation: Evaluate the model's performance using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
5. Deployment: Deploying the model through a user-friendly web application developed using Plotly for interactive visualizations and real-time predictions.






1.	Select the model: 

 Model Selection: The Combination of RNN and LSTM Approach

Overview

To effectively forecast tsunami events, leveraging the strengths of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks can be particularly beneficial. The dataset consists of 16 years of various oceanographic and seismic parameters, which presents a challenge in capturing complex temporal dependencies accurately. 

The RNN and LSTM Combination

Using a combination of RNN and LSTM in the model architecture can help to overcome the limitations of using each model independently. RNNs are capable of capturing temporal dependencies in sequential data but can suffer from the vanishing gradient problem when dealing with long-term dependencies. LSTMs, on the other hand, are designed to address this issue by introducing a memory cell and several gating mechanisms that selectively retain and forget information over long periods (Intelligent Algorithms in Software Engineering, 2020).

The proposed approach is to use an RNN for the initial processing of the data and then use an LSTM for the final prediction. This combination can effectively capture both short-term and long-term dependencies in the data, leading to more accurate predictions (Proceedings of International Conference on Data Science and Applications, 2022).

RNN Layer

The RNN layer is responsible for capturing the temporal dependencies in the input data. The process involves:

1. Sequential Data Input: The input data is fed into the RNN layer one time step at a time. Each time step consists of the input features at that specific point in time.
2. Hidden State Vector: At each time step, the RNN layer transforms the input features into a hidden state vector, which contains information about the current input and previous time steps (Goodfellow, 2016).
3. Passing Hidden State: The hidden state vector is passed to the next time step, where it is updated based on new input features. This process continues until all time steps are processed.
4. Final Hidden State: The final hidden state vector, containing important features that capture temporal dependencies, is output from the RNN layer and passed to the LSTM layer for further processing.

LSTM Layer

The LSTM layer enhances the model's ability to capture long-term dependencies in the data:

1. Feature Concatenation: Additional relevant features for predicting tsunami events are preprocessed and concatenated with the time series data before being fed into the RNN layer.
2. Handling Temporal Dependencies: The RNN layer captures temporal dependencies and extracts important features, which are then fed into the LSTM layer for more accurate predictions (Lipton, 2015).
3. Output from LSTM: The LSTM layer processes the combined input, effectively capturing long-term dependencies and generating predictions based on both the time series data and additional features.

Output Layer

The output layer is responsible for producing the final predictions of tsunami events:

1. Fully Connected Layer: A fully connected (dense) layer with a linear activation function is typically used for regression tasks. The layer takes the output from the LSTM layer as input and produces a vector of predicted values.
2. Optimization: The weights and biases of this layer are optimized during training to minimize the difference between predicted and actual values, as measured by a chosen loss function (Brownlee, 2018).
3. Final Predictions: The output layer generates predictions for the target variable (e.g., tsunami events), providing the final forecast based on the processed data.

Model Design and Training

1. Model Architecture Design: The model architecture is designed by specifying the number and size of the RNN and LSTM layers, activation functions, and other hyperparameters. Experimentation with different architectures and parameters helps in finding the best model for the problem.
2. Data Splitting: The dataset is split into training and test sets to evaluate model performance. The training set is used to fit the model, while the test set assesses the model's performance on out-of-sample data (Ames, 2013).
3. Training Process: The LSTM network is trained using the training set. Optimization algorithms, such as stochastic gradient descent, adjust the weights and biases of the network to minimize error between predicted and actual values.

Hyperparameter Tuning and Model Evaluation

1. Hyperparameter Tuning: Finding the optimal combination of hyperparameters (e.g., learning rate, number of LSTM units, batch size) that maximize model performance.
2. Performance Metrics: Use performance metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to evaluate model accuracy. Cross-validation ensures the model's robustness and stability.

Deployment

1. Web Application Development: Develop a user-friendly web application using Plotly for interactive visualizations and real-time tsunami event predictions.
2. Model Integration: Integrate the trained LSTM RNN model into the web application to process new data inputs and generate real-time predictions.
3. User Access: Provide access to disaster management agencies, coastal communities, and other stakeholders for making informed decisions based on predictions.

Informative Analysis and Insights

The insights from the correlation matrix and data preprocessing guide feature selection and model development. Key relationships identified help in understanding underlying patterns, leading to accurate predictions.

By leveraging the strengths of RNNs and LSTMs, the proposed system aims to provide accurate and reliable tsunami forecasts, enhancing early warning systems and preparedness for coastal communities.







B.	Testing & Problems Encountered

 1. Evaluate the LSTM
After training the LSTM model, its performance is evaluated on the test set to gauge its predictive accuracy on unseen data. Performance metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used to quantify the model’s accuracy and effectiveness in making predictions. This evaluation helps in understanding how well the model generalizes to new, unseen data and identifies areas where the model may need improvement.

 2. Fine-tune the LSTM
If the initial performance of the LSTM model is not satisfactory, fine-tuning is required. This involves adjusting the hyperparameters, such as learning rate, number of LSTM units, batch size, and number of layers. Additionally, modifications to the network architecture may be necessary. This iterative process includes re-evaluating the model’s design and training procedures with different configurations to enhance its performance and achieve optimal predictive accuracy.

 C. Results

Training an LSTM model is a time-intensive process that often requires multiple iterations to identify the most effective configuration. Continuous monitoring of the model’s performance over time is crucial to ensure that it maintains its predictive accuracy. Regular evaluations help in detecting any performance degradation and necessitate periodic fine-tuning to adapt to new data patterns and maintain reliable predictions.

 D. Data Visualization Web App

Data visualization plays a pivotal role in data analysis and communication by transforming complex datasets into comprehensible visual formats. With the proliferation of data and the imperative to make data-driven decisions, there is an increasing demand for interactive and intuitive web-based applications for real-time data exploration and visualization.

 Developing the Visualization App with Plotly
Plotly is a robust open-source library for creating interactive and dynamic visualizations in web browsers. This research aims to develop a data visualization web application using Plotly, enabling users to easily visualize and interact with various types of data.

The application will feature a user-friendly interface for data manipulation and offer a range of customizable visualization options. The integration of machine learning techniques will allow the generation of insights and predictions from the data, which can be incorporated into the application for advanced data analysis. The ultimate goal is to create an interactive web-based tool that empowers users to derive insights and make informed decisions based on their data.

 Key Features of the Web App:
- User-Friendly Interface: The application will be designed for ease of use, allowing users to navigate and manipulate data seamlessly.
- Customizable Visualizations: Users will have access to a variety of visualization options that can be tailored to meet their specific needs.
- Real-Time Interaction: The app will support real-time data interaction, enabling users to explore and analyze data dynamically.
- Machine Learning Integration: Advanced analytics and predictive insights will be provided through the incorporation of machine learning models.

By leveraging Plotly and machine learning, the data visualization web app aims to be a powerful tool for disaster management agencies, coastal communities, and other stakeholders. It will facilitate better understanding and decision-making based on the predictive analysis of tsunami events and other related data.
  






References

Advances In Electrical And Computer Technologies. (2021).
Aldi, M. W. P., Jondri, J., & Aditsania, A. (2018). Analisis Dan Implementasi Long. Retrieved From Https://Openlibrarypublications.Telkomuniversity.Ac.Id/Index.Php/Engineeri
Ames, G. W. (2013). An Introduction To Statistical Learning (Vol. 112). New York: Springer.
Bashiri, B. M.-K. (2021). Forecasting Inflation In Iran Using Hybrid Models: A Comparison Between Machine Learning And Time Series Models. Journal Of Economic Studies.
Bontempi, G. B. (2019). Machine Learning Strategies For Time Series Forecasting. Frontiers In Artificial Intelligence, Https://Doi.Org/10.3389/Frai.2019.00004.
Brownlee, J. (2018). How To Develop LSTM Models For Time Series Forecasting. Machine Learning Mastery.
Goodfellow, I. B. (2016). Deep Learning (Chapter 10: Sequence Modeling: Recurrent And Recursive Nets). MIT Press.
Inc, P. T. (2021). Plotly: The Front-End For ML And Data Science Models. Https://Plotly.Com/.
Intelligent Algorithms In Software Engineering. (2020).
James, G. W. (2013). An Introduction To Statistical Learning. New York: Springer.
Lee, K. H. (2020). Forecasting Inflation Rates In Korea Using Deep Learning Models. Sustainability, 2048.
Lipton, Z. C. (2015). Critical Review Of Recurrent Neural Networks For Sequence Learning. Arxiv Preprint Arxiv:1506.00019.
Machine Learning Algorithms. (2018).
Martin T. Hagan, H. B. (2014). Neural Network Design. 
Nakamura, E. (2005). Inflation Forecasting Using A Neural Network. Economics Letters.
Proceedings Of International Conference On Data Science And Applications. (2022).
Roser, M. A.-O. (2021). Machine Learning. Ourworldindata.Org.
Warjiyo, P. (2019). CENTRAL BANK POLICY: THEORY AND PRACTICE. Jakarta: Bank Indonesia Institute.
Zhang, G. &. (2019). Ong Short-Term Memory Neural Network For Air Pollution Prediction: Method Development And Evaluation. Environmental Pollution.

