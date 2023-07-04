import pickle as pkl
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

compute_path = workspace + '/compute_estimates.pkl'  # Replace with the desired file path
memory_path = workspace + '/memory_estimates.pkl'  # Replace with the desired file path

# Load the compute data from pickle file
with open(compute_path, 'rb') as file:
    compute_data = pkl.load(file)

# Load the memory data from pickle file
with open(memory_path, 'rb') as file:
    memory_data = pkl.load(file)

# Extract x and y values from dictionaries
x = list(compute_data.values())
y = list(memory_data.values())
labels = list(compute_data.keys())  # Use keys as labels

# Create the graph
fig = go.Figure()

# Plotting the scatter plot
for i in range(len(x)):
    fig.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode='markers', name=labels[i],marker=dict(size=15, symbol=i)))

# # Plotting the scatter plot
# fig.add_trace(go.Scatter(x=x, y=y, mode='markers', text=labels))

# Customize the layout
fig.update_layout(
    title={
        'text': 'Compute (FLOPs) vs Memory Bandwidth',
        'x': 0.5,  # Align title to the center horizontally
    },
    xaxis_title='Compute (FLOPs)',
    yaxis_title='Memory Bandwidth/Inference',
    xaxis_type='log',  # Set x-axis to log scale
    showlegend=True,  # Display legend
)

# Display the graph
fig.show()
fig.write_image('compute_vs_bandwidth.jpg')
fig.write_html('image.html')
