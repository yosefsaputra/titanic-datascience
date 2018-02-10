from plotly.plotly import plot
import plotly.graph_objs as go

import random

x_data = range(0, 10)
y_data = [random.randint(0, 10) for i in x_data]

# Edit the plot color using marker
plots = [
    go.Bar(x=x_data,
           y=y_data,
           name='bar_chart',
           marker={'color': '#fa8072'}),        # salmon color
    go.Scatter(x=x_data,
               y=y_data,
               name='line_chart',
               marker={'color': '#9b30ff'}),    # purple
]

# Create Layout object
layout = go.Layout()
# Set plot title
layout['title'] = 'My Simple Plot'
# Set x-axis properties
layout['xaxis']['title'] = 'x_data (integer)'   # add x-axis title
layout['xaxis']['color'] = '#ee2c2c'            # red
layout['xaxis']['showgrid'] = True
# Set y-axis properties
layout['yaxis']['title'] = 'y_data (integer)'   # add y-axis title
layout['yaxis']['color'] = '#1874cd'            # blue
layout['yaxis']['showgrid'] = True

# Create Figure object
figure = go.Figure(data=plots, layout=layout)

# Plot the Figure object
plot(figure, filename='myfirstplot')
