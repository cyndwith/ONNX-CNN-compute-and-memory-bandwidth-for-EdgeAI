import plotly.graph_objects as go
from plotly.offline import plot
# Extract keys and values from the dictionary
categories = list(compute_estimate.keys())
counts = list(compute_estimate.values())

# Create a bar trace for the histogram
trace = go.Bar(x=categories, y=counts)

# Create the layout for the histogram
layout = go.Layout(title={'text': 'Network Architecture vs FLOPs',
                          'x': 0.5,  # Align title to the center horizontally
                   },
                   xaxis=dict(title='Network Architecture'),
                   yaxis=dict(title='FLOPs',
                   type='log'))

# Create the figure and add the trace
fig = go.Figure(data=[trace], layout=layout)

# Show the figure
fig.show()
fig.write_image('compute.jpg')
fig.write_html('image.html')
