import plotly.graph_objects as go

# Extract keys and values from the dictionary
categories = list(memory_estimate.keys())
counts = list(memory_estimate.values())

# Create a bar trace for the histogram
trace = go.Bar(x=categories, y=counts)

# C reate the layout for the histogram
layout = go.Layout(title= {
                      'text': 'Network Architecture vs Memory B/W',
                      'x': 0.5,
                   },
                   xaxis=dict(title='Network Architecture'),
                   yaxis=dict(title='Memory B/W'))

# Create the figure and add the trace
fig = go.Figure(data=[trace], layout=layout)

# Show the figure
fig.show()
fig.write_image('bandwidth.jpg')
fig.write_html('image.html')
