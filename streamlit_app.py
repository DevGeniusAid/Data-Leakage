# In this notebook, we explore different types of data leakage. Data leakage can deceive us into thinking our model performs well on both training and validation datasets, but it often fails to generalize to new data. This occurs when the model inadvertently learns from information it shouldn't have access to, leading to overfitting and reduced predictive ability. In our scenario, the test dataset was generated as a random sample from the training dataset.

# +
import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from streamlit_folium import st_folium
from streamlit_navigation_bar import st_navbar
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly.subplots import make_subplots

#Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(classification_report, confusion_matrix, fbeta_score,
                           make_scorer, recall_score)

from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# %pip install imbalanced-learn
import warnings
warnings.filterwarnings('ignore')
# -

train_data = pd.read_csv('train.csv', sep=';')
test_data = pd.read_csv('test.csv', sep=';')
pd.set_option('display.max_columns', None)

st.set_page_config(layout='wide')

# **We will use streamlit to have a nice dashboard or report**

tab1, tab2 = st.tabs(['Data leakage analysis', 'ML Model'])

# Data transformation
train_data.info()

# Observations:
#
# 452111 rows
# 17 Variables
# 7 as int64
# 10 as objects.
# We will change the objects data type to category, this will save a lot of memory.

# Describe data
train_data.describe()

train_data.describe(include='object')

# Objects to Categorical
train_data = train_data.astype({'job' : 'category', 'marital':'category', 'education':'category',
                               'default':'category', 'housing':'category', 'loan':'category', 'contact':'category', 'month':'category', 'poutcome':'category', 'y':'category'})

train_data.info()

# Record the target variable (no = 0 yes = 1)
train_data['y'] = np.where(train_data['y'] == 'no', 0,1)
test_data['y'] = np.where(test_data['y'] == 'no', 0,1)

# **Visualizations: Job, Housing, Marital and Education Distributions**

# +

with tab1:
    # Function to generate counts and bar plots for a given column
    def counts_plot(y_var, col):
        # Generate counts and percentages
        y_var_counts = (
            train_data[y_var].dropna()
            .value_counts()
            .reset_index()
            .rename(columns={"index": y_var, y_var: "counts"})
            .assign(
                percent=lambda df_: (df_["counts"] / df_["counts"].sum()).round(2) * 100
            )
            .sort_values(by="counts", ascending=True)
        )

        # Create bar plot using Plotly Graph Objects
        bar = go.Bar(
            x=y_var_counts['percent'],
            y=y_var_counts[y_var],
            orientation='h',
            text=y_var_counts['percent'],
            texttemplate='%{text:.2s}%',
            marker=dict(color=col),
            name=str.title(y_var)
        )
        return bar

    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Job", "Housing", "Marital", "Education"),
        shared_xaxes=False,  # Independent axes for each plot
        vertical_spacing=0.1
    )

    # Add subplots for each category
    fig.add_trace(counts_plot("job", col="#8da0cb"), row=1, col=1)
    fig.add_trace(counts_plot("housing", col="#e78ac3"), row=1, col=2)
    fig.add_trace(counts_plot("marital", col="#a6d854"), row=2, col=1)
    fig.add_trace(counts_plot("education", col="#ffd92f"), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=600,
        title_text="Subplots for Job, Housing, Marital, and Education",
        showlegend=False
    )

    # Show the figure
    st.plotly_chart(fig)
# -

# **Visualizations: Default, Loan, Contact and Poutcome Distributions**

with tab1:
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Default", "Loan", "Contact", "Poutcome Distributions"),
        shared_xaxes=False,  # Independent axes for each plot
        vertical_spacing=0.1
    )

    # Add subplots for each category
    fig.add_trace(counts_plot("default", col="#8da0cb"), row=1, col=1)
    fig.add_trace(counts_plot("loan", col="#e78ac3"), row=1, col=2)
    fig.add_trace(counts_plot("contact", col="#a6d854"), row=2, col=1)
    fig.add_trace(counts_plot("poutcome", col="#ffd92f"), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=600,
        title_text="Subplots for Default, Loan, Contact, Poutcome Distributions",
        showlegend=False
    )

    # Show the figure
    st.plotly_chart(fig)

# **Target variable Counts**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nTarget variable Counts</h3>", unsafe_allow_html=True)
    target_v = (
           train_data.loc[:, 'y']
           .value_counts()
           .reset_index()
           .rename(columns={"index": "y", "y": "counts"})
           .assign(percent=lambda df_: (df_["counts"] / df_["counts"].sum()).round(2) * 100)
    )
    target_v

with tab1:
    # Prepare the data
    y_var = (
        train_data.loc[:, "y"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "y", "y": "counts"})
        .assign(percent=lambda df_: (df_["counts"] / df_["counts"].sum()).round(2) * 100)
    )

    # Define labels and colors
    labels = ["0 | Not subscribed", "1 | Subscribed"]
    target_color = ["#e42256", "#00b1b0"]  # Colors for the pie slices

    # Create a pie chart using Plotly
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=y_var["percent"],  # Percentage data
                textinfo="label+percent",  # Show both label and percentage
                insidetextorientation="radial",  # Make sure text is well positioned
                marker=dict(colors=target_color),  # Set colors for the slices
                hole=0.0,  # Full pie chart (no donut shape)
                pull=[0.1, 0],  # "Explode" effect for the first slice
                hoverinfo="label+percent+value"  # Show extra info on hover
            )
        ]
    )

    # Add title and annotations
    fig.update_layout(
        title_text="Target Variable Proportions",
        annotations=[
            dict(
                text=f"count: {y_var.iloc[0, 1]}",
                x=0.2, y=0.8, showarrow=False, font_size=12, font_color="black"
            ),
            dict(
                text=f"count: {y_var.iloc[1, 1]}",
                x=-0.2, y=-0.7, showarrow=False, font_size=12, font_color="black"
            )
        ]
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)
    st.write("Unbalanced Target Variable!!")

# **Numerical features(pairplot**

with tab1:
    # Select numerical features from the dataset
    Num_feats = train_data.select_dtypes("integer").copy()

    # Sample a fraction (1%) of the data from each class in 'y'
    sample_for_pair_plot = Num_feats.groupby("y", group_keys=False).apply(
        lambda x: x.sample(frac=0.01)
    )

    # Define colors for the target variable
    target_color = ["#e42256", "#00b1b0"]  # Colors for the different classes of 'y'

    # Create a scatter matrix using Plotly Express
    fig = px.scatter_matrix(
        sample_for_pair_plot,  # The sampled data
        dimensions=Num_feats.columns,  # Numerical features for pair plot
        color="y",  # Use 'y' for color coding
        title="Scatter Matrix for Numerical Features (Sampled)",
        labels={col: col for col in Num_feats.columns},  # Axis labels
        color_continuous_scale=target_color,  # Color palette for target classes
        height=700  # Adjust height
    )

    # Update marker size and layout for better visualization
    fig.update_traces(marker=dict(size=3, opacity=0.7))  # Set marker size
    fig.update_layout(showlegend=True)  # Optionally show the legend

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

# **Age, Duration, Balance and Campaign Variables**<br>
# Creating a Function

with tab1:
    from scipy.stats import gaussian_kde
    def num_distributions(var_1, var_2):
        # Prepare the data
        age_dur = train_data[[var_1, var_2, "y"]]

        # Define colors for the target variable
        target_color = ["#e42256", "#00b1b0"]  # Replace with your color palette if needed

        # Initialize lists for histogram, KDE, and line traces
        hist_data = []
        kde_data = []
        line_data = []

        for label, color in zip(age_dur["y"].unique(), target_color):
            subset = age_dur[age_dur["y"] == label]

            # Create histogram trace
            hist_trace = go.Histogram(
                x=subset[var_1],
                name=f"Subscribed: {label}",
                marker_color=color,
                opacity=0.6,
                nbinsx=20,
                histnorm='',
                showlegend=True
            )
            hist_data.append(hist_trace)

            # Compute KDE
            kde = gaussian_kde(subset[var_1])
            x = np.linspace(subset[var_1].min(), subset[var_1].max(), 100)
            kde_line = kde(x)

            # Create KDE trace
            kde_trace = go.Scatter(
                x=x,
                y=kde_line,
                mode='lines',
                name=f"KDE: {label}",
                line=dict(color=color, width=1.5, dash='dash'),
                showlegend=False
            )
            kde_data.append(kde_trace)

            # Calculate mean and add vertical line
            mean_value = subset[var_1].mean()
            line_trace = go.Scatter(
                x=[mean_value, mean_value],
                y=[0, subset[var_1].max()],
                mode='lines',
                name=f"Mean: {label}",
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            )
            line_data.append(line_trace)

        # Create scatter plot
        scatter_fig = px.scatter(
            age_dur,
            x=var_1,
            y=var_2,
            color="y",
            opacity=0.6,
            color_discrete_sequence=target_color,  # Color for the scatter plot
            title=f"{var_2.title()} Distributions"
        )

        # Update layout and styling for the scatter plot
        scatter_fig.update_layout(
            xaxis_title=var_1.title(),
            yaxis_title=var_2.title(),
            legend_title="Subscribed?"
        )

        # Create subplot figure with two plots side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{var_1.title()} Distributions", f"{var_2.title()} Distributions"],
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )

        # Add histogram, KDE, and line traces to the first subplot
        fig.add_traces(hist_data + kde_data + line_data, rows=1, cols=1)

        # Add scatter plot to the second subplot
        for trace in scatter_fig.data:
            fig.add_trace(trace, row=1, col=2)

        # Update layout for the combined figure
        fig.update_layout(
            title_text=f"Distributions of {var_1.title()} and {var_2.title()}",
            height=600,
            showlegend=True
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig)

with tab1:
    # 
    num_distributions("age", "duration")
    num_distributions("balance", "duration")

# **Who talks more**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nWho talks more</h3>", unsafe_allow_html=True)
    larg_dur = train_data["duration"].nlargest(10)
    small_dur = train_data["duration"].nsmallest(5)
    filtered_data = train_data.query("duration in @larg_dur | duration in @small_dur").sort_values(
        by="duration", ascending=False
    )
    
    # Display the styled DataFrame
    st.dataframe(filtered_data.style.background_gradient())
    st.write('There is a call subject that took about 5000 Sec (1.30 hours), which is pretty lenghty but it might be data entry issues because we do have call records with 0 and 1 sec. duration using a cellular networl! and it might be the sum of many call attempts, so we need more analysis to make a proper decision or even a correct assumption!')

# **Default and target variable "y"**

with tab1:
    st.markdown("<h3 style='text-align: center;'>\nDefault and target variable 'y'</h3>", unsafe_allow_html=True)
    target = (
                train_data[['default', 'y']]
                .value_counts()
                .reset_index()
                .rename(columns={0: "counts"})
                .style.background_gradient()
            )
    target

with tab1:
    target1 = (
                train_data[['default', 'y']]
                .value_counts()
                .reset_index()
                .rename(columns={0: "counts"})
               
            )

with tab1:
    # Create a figure with subplots
    st.markdown("<h3 style='text-align: center;'>\nDefault distributions vs Target (y)</h3>", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Unique values in 'y' column
    unique_y = target1['y'].unique()

    # Add a bar trace for each unique value in 'y'
    for val in unique_y:
        filtered_data = target1[target1['y'] == val]
        fig.add_trace(
            go.Bar(x=filtered_data['default'], y=filtered_data['counts'], name=str(val)),
            row=1, col=1
        )

    # Update layout for the Plotly figure
    fig.update_layout(
        yaxis_title="Counts",
        barmode='group',  # Ensure bars are side by side
        legend=dict(title="Subscribed?", orientation="h", yanchor="top", xanchor="center", x=0.5),
    )

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig)

with tab1:
    target_3 = (
        train_data[["balance", "default", "y"]]
        .groupby(["default", "y"])["balance"]
        .agg(["mean", "count"])
        .reset_index()
        .style.background_gradient()
        )
    target_3

# ------------------------

# ### Machine learning<br>
# **Model**

with tab2:
    st.markdown("<h3 style='text-align: center;'>\nModel</h3>", unsafe_allow_html=True)
    rf_seed = 345
    np.random.seed(rf_seed)
    
    train_x = train_data.iloc[:, :-1]
    see = train_x.head()
    
    train_y = train_data[['y']]
    train_y = np.ravel(train_y)
    see2 = train_y.shape
    
    see
    st.write("Shape: ",see2)


# **Encoding**

with tab2:
    ohe_columns = list(train_x.select_dtypes(include='category').columns.values)
    st.write("Encoding: ",ohe_columns)
    st.write("I will remove the education variable for it is to be encoded as an ordinal type!")
    
    ohe_columns.remove("education")
    st.write("Updated Encoding: ", ohe_columns)
    
    #lets get dummies
    train_x = pd.get_dummies(
            train_x, prefix=ohe_columns, columns=ohe_columns, drop_first=True
            )
    st.write(train_x.head())

with tab2:
    recode_education_var = {"unknown":0, "primary":1, "secondary":2, "tertiary":3}
    train_x['education'] = train_x['education'].replace(recode_education_var)
    
    st.write("Record education var: ",train_x['education'].value_counts(normalize=True))
    st.write("train y shape: ", train_y.shape)
    
    test_x = test_data.iloc[:, :-1]
    st.write("test x shape:", test_x.shape)
    
    test_y = test_data[["y"]]
    test_y = np.ravel(test_y)
#     st.write("test y: ",test_y.shape)
    
    test_x = pd.get_dummies(
    test_x, prefix=ohe_columns, columns=ohe_columns, drop_first=True
    )
#     st.write("test x:",test_x.shape)

    test_x["education"] = test_x["education"].replace(recode_education_var)
    st.write("test x:",test_x.shape)
    st.write("test y:",test_y.shape)

# **Random Forest Model**

with tab2:
    rf = RandomForestClassifier(
        n_jobs=-1, random_state=rf_seed, class_weight="balanced_subsample")
    
    # recall_score
    rs = make_scorer(recall_score)
    
    # CV
    cv = cross_val_score(rf, train_x, train_y, cv=10, n_jobs=-1, scoring=rs)
    st.write("Cross validation scores: {}".format(cv))
    st.write("%0.2f recall with a standard deviation of %0.2f" % (cv.mean(), cv.std()))

with tab2:
    #fit the model
    see = rf.fit(train_x, train_y)
    st.write("fit the model:",see)
    
    #recall score
    pred = rf.predict(train_x)
    st.write("The train recall score is {}".format(np.round(recall_score(train_y, pred), 4)))

# +
# Compute confusion matrix
cm = confusion_matrix(train_y, pred)

# Create a Plotly heatmap
fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted 0', 'Predicted 1'],
    y=['Actual 0', 'Actual 1'],        
    colorscale='Greens',
    showscale=False,
    text=cm, 
    texttemplate="%{text}",
    textfont={"size": 15}
))

# Update layout for title, labels, etc.
fig.update_layout(
    title="Confusion Matrix (Train Set)",
    xaxis_title="Predicted",
    yaxis_title="Actual",
    font=dict(size=16)
)
print(classification_report(train_y, pred))

# Display the Plotly figure in Streamlit
with tab2:
    st.plotly_chart(fig)
    st.write(classification_report(train_y, pred))

# -

with tab2:
    pred = rf.predict(test_x)
    st.write("The test recall score is: {}".format(np.round(recall_score(test_y, pred), 4)))

with tab2:
        # Compute confusion matrix for test set
    cm_test = confusion_matrix(test_y, pred)

    # Create a Plotly heatmap for the test set
    fig_test = go.Figure(data=go.Heatmap(
        z=cm_test,
        x=['Predicted 0', 'Predicted 1'],  # Adjust labels based on your classes
        y=['Actual 0', 'Actual 1'],        # Adjust labels based on your classes
        colorscale='Greens',
        showscale=False,
        text=cm_test, 
        texttemplate="%{text}",
        textfont={"size": 15}
    ))

    # Update layout for title, labels, etc.
    fig_test.update_layout(
        title="Confusion Matrix (Test Set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(size=16)
    )
    print(classification_report(test_y, pred))

    # Display the Plotly figure
    st.plotly_chart(fig_test)

    # Print classification report
    st.write(classification_report(test_y, pred))
    st.write("Despite having an imbalanced target variable, we didn’t apply any techniques like SMOTE to address it. All the metrics are showing 100%, which raises a significant concern about potential data leakage. It’s important to remember that the dataset's description explicitly mentions that the test data was randomly drawn from the training data.The dataset description states:'test.csv: 4521 rows and 18 columns with 10% of the examples (4521), randomly selected from train.csv.'As a result, we can't rely on this final outcome, as it's likely that the high performance is due to data leakage. However, the model's accuracy on the training data could be somewhat justified.To address this, we should split the original training data into separate training and testing sets and then re-evaluate the results.")


# **Random Forest Model 2 (Splitting the training data)**

with tab2:
    st.markdown("<h3 style='text-align: center;'>\nRandom Forest Model 2 (Splitting the training data)</h3>", unsafe_allow_html=True)
    X = train_data.iloc[:, :-1]
    st.write(X.head())
    
    y = train_data[["y"]]

    y = np.ravel(train_y)
    st.write("shape:y.shape")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify= y, random_state=rf_seed)
    st.write("X train:",X_train.shape)
    st.write("Y train:",y_train.shape)
    st.write("X test",X_test.shape)
    st.write("Y",y_test.shape)

# **Random Forest Model 2, Encoding 2**

with tab2:
    st.markdown("<h3 style='text-align: center;'>\nRandom Forest Model 2, Encoding 2</h3>", unsafe_allow_html=True)
    ohe_columns = list(X_train.select_dtypes(include='category').columns.values)
    ohe_columns.remove('education')
    st.write(ohe_columns)
    
    X_train = pd.get_dummies(
    X_train, prefix=ohe_columns, columns=ohe_columns, drop_first=True
    )
    st.write(X_train.head())
    
    recode_education_var = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    X_train["education"] = X_train["education"].replace(recode_education_var)
    st.write("Value counts",X_train["education"].value_counts(normalize=True))
    
    st.write("y train shape:", y_train.shape)
    X_test = test_data.iloc[:, :-1]
    st.write("x train shape: ", X_test.shape)
    
    y_test = test_data[["y"]]
    y_test = np.ravel(test_y)
    st.write("ravel: ",y_test)
    
    X_test = pd.get_dummies(
    X_test, prefix=ohe_columns, columns=ohe_columns, drop_first=True
    )
    st.write("X test: ",X_test.shape)
    
    X_test["education"] = test_x["education"].replace(recode_education_var)
    st.write("X test: ",X_test.shape)
    st.write("y test: ",y_test.shape)

with tab2:
    rf2 = RandomForestClassifier(
    n_jobs=-1, random_state=rf_seed, class_weight="balanced_subsample"
    )
    # recall_score
    rs2 = make_scorer(recall_score)

    #Cross-validation for recall score
    cv = cross_val_score(rf2, train_x, train_y, cv=10, n_jobs=-1, scoring=rs2)
    st.write(f"Cross-validation recall scores: {cv}")
    st.write(f"Mean recall: {cv.mean():.2f} with a standard deviation of {cv.std():.2f}")

    # Fit the model
    rf2.fit(X_train, y_train)

    # Train set predictions
    train_pred = rf2.predict(X_train)
    train_recall = np.round(recall_score(y_train, train_pred), 2)
    st.write(f"The train recall score is {train_recall}")

    # Confusion matrix for the train set
    train_cm = confusion_matrix(y_train, train_pred)
    fig_train = px.imshow(train_cm, text_auto=True, color_continuous_scale="Greens")
    fig_train.update_layout(
        title="Confusion Matrix (Train Set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    st.plotly_chart(fig_train)

    # Display classification report for the train set
    st.write("Classification report (Train Set):")
    st.text(classification_report(y_train, train_pred))

    # Test set predictions
    test_pred = rf2.predict(X_test)
    test_recall = np.round(recall_score(y_test, test_pred), 2)
    st.write(f"The test recall score is {test_recall}")

    # Confusion matrix for the test set
    test_cm = confusion_matrix(y_test, test_pred)
    fig_test = px.imshow(test_cm, text_auto=True, color_continuous_scale="Greens")
    fig_test.update_layout(
        title="Confusion Matrix (Test Set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    st.plotly_chart(fig_test)

    # Display classification report for the test set
    st.write("Classification report (Test Set):")
    st.text(classification_report(y_test, test_pred))
    st.write("That outcome seems reasonable, but I still have some reservations! 😄It's also crucial to keep track of how the model performs on unseen data. This is essential for spotting possible data leakage issues and ensuring the model generalizes effectively to new, unseen datasets.Thank you for taking the time to read this. I'd greatly appreciate any feedback or insights you might have on this topic!")





# !streamlit run streamlit_app.py


