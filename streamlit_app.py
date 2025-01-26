import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Interactive Linear Regression Calculator")
st.markdown("""
### Features:
1. Enter your **x** and **y** values below (comma-separated).
2. View the regression model's slope and intercept.
3. Visualize the regression line.
4. Predict **y** for a given **x** value.
""")

# Input Section
st.header("Feature 1: Input Your Data")

if "x_values" not in st.session_state:
    st.session_state.x_values = ""
if "y_values" not in st.session_state:
    st.session_state.y_values = ""

x_values = st.text_input("Enter x values (comma-separated):", st.session_state.x_values)
y_values = st.text_input("Enter y values (comma-separated):", st.session_state.y_values)

if st.button("Submit Data"):
    st.session_state.x_values = x_values
    st.session_state.y_values = y_values

try:
    # Process Inputs
    x_values = list(map(float, x_values.split(',')))
    y_values = list(map(float, y_values.split(',')))

    if len(x_values) != len(y_values):
        st.error("Number of x and y values must be the same.")
    else:
        # Convert to numpy arrays
        x = np.array(x_values).reshape(-1, 1)
        y = np.array(y_values)

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)

        slope = model.coef_[0]
        intercept = model.intercept_

        st.success("Regression Model Calculated!")
        st.write(f"### Slope (β1): `{slope:.4f}`")
        st.write(f"### Intercept (β0): `{intercept:.4f}`")

        # Plot the data and regression line
        st.header("Feature 2: Visualize the Regression")
        fig, ax = plt.subplots()
        ax.scatter(x, y, color="blue", label="Data Points")
        ax.plot(x, slope * x + intercept, color="red", label="Regression Line")
        ax.set_title("Linear Regression")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        st.pyplot(fig)

        # Prediction Section
        st.header("Feature 3: Predict y for a Given x")
        x_input = st.number_input("Enter an x value for prediction:", step=0.1)

        if st.button("Predict y"):
            y_pred = model.predict([[x_input]])[0]
            st.write(f"### Predicted y: `{y_pred:.4f}`")

except ValueError:
    if x_values or y_values:
        st.error("Please enter valid numeric values separated by commas.")
