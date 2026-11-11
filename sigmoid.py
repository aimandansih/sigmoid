
import streamlit as st
import numpy as np
import pandas as pd

# Streamlit page configuration
st.set_page_config(
    page_title="Sigmoid Activation Function",
    layout="wide"
)

st.title("Sigmoid Activation Function")
st.write("Visualisation of the Sigmoid activation function.")

# Generate input values
x = np.linspace(-6, 6, 400)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)

# Display formula
st.subheader("Mathematical Formula")
st.latex(r"f(x) = \frac{1}{1+e^{-x}}")

# Create DataFrame
df = pd.DataFrame({
    "Input (x)": x,
    "Sigmoid Output": y
})

# Plot graph
st.subheader("Sigmoid Activation Function Graph")

st.line_chart(
    df.set_index("Input (x)")
)

# Explanation
st.subheader("Explanation")

st.write("""
Sigmoid converts input values into an output range
between 0 and 1.

It produces an S-shaped curve and is commonly used
for binary classification problems.
""")
