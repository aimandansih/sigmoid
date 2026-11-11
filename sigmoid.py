import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function")

st.title("Sigmoid Activation Function")
st.write("Visualisation of the Sigmoid function.")

# Generate input values
x = np.linspace(-6, 6, 400)

# Sigmoid function
y = 1 / (1 + np.exp(-x))

# Plot graph
fig, ax = plt.subplots()
ax.plot(x, y, label="Sigmoid", linewidth=2)

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

ax.set_title("Sigmoid Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.write("Formula: f(x) = 1 / (1 + exp(-x))")