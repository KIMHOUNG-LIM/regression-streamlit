import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset using seaborn
iris = sns.load_dataset('iris')
st.write("Dataset Overview:")
st.write(iris.head())

# Streamlit app
st.title("Model Deployment")

st.sidebar.subheader("Model Deployment Streamlit")
# Example tabs for navigation
tab = st.sidebar.radio(
    "Iris Dataset",
     ['Multi Linear Regression', 'Logistic Regression']
)

# # Sidebar navigation
# tab = st.sidebar.selectbox('Select a model', ['Multi Linear Regression', 'Logistic Regression'])


if tab == 'Multi Linear Regression':
    st.header("Multi Linear Regression")
    
    # Encode the 'species' column using LabelEncoder
    encoder = LabelEncoder()
    iris['species'] = encoder.fit_transform(iris['species'])

    X = iris[['petal_length', 'sepal_length', 'species']]
    y = iris['petal_width'] 
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression().fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # st.write(f"Mean Squared Error: {mse}")
    # st.write(f"R-squared: {r2}")

    # # Model coefficients and intercept
    # st.write(f"Coefficients: {model.coef_}")
    # st.write(f"Intercept: {model.intercept_}")

    # # Equation of the model
    # equation = f"y = {model.coef_[0]:.6f}*petal_length + {model.coef_[1]:.6f}*sepal_length + {model.coef_[2]:.6f}*species_encoded + {model.intercept_:.6f}"
    # st.write(f"Model Equation: {equation}")

    # Create sliders for interactive input
    petal_length = st.slider("Select Petal Length", min_value=float(X['petal_length'].min()), max_value=float(X['petal_length'].max()), value=5.0, step=0.1)
    sepal_length = st.slider("Select Sepal Length", min_value=float(X['sepal_length'].min()), max_value=float(X['sepal_length'].max()), value=5.0, step=0.1)
    
    # Species selection with slider (as a numeric value)
    species = st.slider("Select Species (0 = Setosa, 1 = Versicolor, 2 = Virginica)", min_value=0, max_value=2, value=0, step=1)
    
    # Prepare the input features as a numpy array
    input_features = np.array([[petal_length, sepal_length, species]])

    # Predict the Petal Width for the input values
    predicted_petal_width = model.predict(input_features)

    # Display predicted result
    st.write(f"Predicted Petal Width: {predicted_petal_width[0]:.4f}")

    # Seaborn scatter plot for Actual vs Predicted values
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Petal Width')
    plt.ylabel('Predicted Petal Width')
    plt.title('Actual vs Predicted Petal Width')

    # Regression line (trendline)
    z = np.polyfit(y_test, y_pred, 1)  # Fit a line (y = mx + b)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='red', linestyle='--', label='Trendline')
    plt.legend()
    
    # Create a figure and pass it to st.pyplot
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)

    # Residual plot function
    def residual_plot(y_test, y_pred):
        # Calculate residuals
        residuals = y_test - y_pred

        # Create a figure with subplots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot of residuals vs. predicted values
        ax[0].scatter(y_pred, residuals, color='blue', alpha=0.6, edgecolor='k', s=50)
        ax[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax[0].set_xlabel('Predicted Petal Width', fontsize=12)
        ax[0].set_ylabel('Residuals', fontsize=12)
        ax[0].set_title('Residual Plot', fontsize=14)
        ax[0].grid(alpha=0.3)

        # Histogram of residuals
        sns.histplot(residuals, bins=20, kde=True, color='blue', ax=ax[1], alpha=0.6)
        ax[1].set_xlabel('Residuals', fontsize=12)
        ax[1].set_ylabel('Frequency', fontsize=12)
        ax[1].set_title('Residual Distribution', fontsize=14)
        ax[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Call the residual plot function
    residual_plot(y_test, y_pred)


if tab == 'Logistic Regression':
    st.header("Logistic Regression")
    
    # Prepare input features and binary target
    X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = (iris['species'] == 'setosa').astype(int)  # Binary target: 1 if setosa, 0 otherwise
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression(max_iter=200).fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    # Model coefficients and intercept
    st.write(f"Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")

    # Create sliders for interactive input
    sepal_length = st.slider("Select Sepal Length", min_value=float(iris['sepal_length'].min()), max_value=float(iris['sepal_length'].max()), value=5.0, step=0.1)
    sepal_width = st.slider("Select Sepal Width", min_value=float(iris['sepal_width'].min()), max_value=float(iris['sepal_width'].max()), value=3.0, step=0.1)
    petal_length = st.slider("Select Petal Length", min_value=float(iris['petal_length'].min()), max_value=float(iris['petal_length'].max()), value=1.5, step=0.1)
    petal_width = st.slider("Select Petal Width", min_value=float(iris['petal_width'].min()), max_value=float(iris['petal_width'].max()), value=0.2, step=0.1)
    
    # Prepare the input features as a numpy array
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the Species for the input values
    predicted_class = model.predict(input_features)
    st.write(f"Predicted Class (1 = Setosa, 0 = Not Setosa): {predicted_class[0]}")

    # Visualize the decision boundaries (for 2 features only)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Select two features for visualization
    x_min, x_max = iris['sepal_length'].min() - 1, iris['sepal_length'].max() + 1
    y_min, y_max = iris['petal_length'].min() - 1, iris['petal_length'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(iris['sepal_length'], iris['petal_length'], c=y, edgecolors='k', marker='o', s=80, cmap=plt.cm.Paired)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Petal Length')
    ax.set_title('Decision Boundaries for Logistic Regression')

    st.pyplot(fig)
