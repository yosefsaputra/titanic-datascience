from plotly.plotly import plot
import plotly.graph_objs as go
import pandas as pd

country_area = pd.read_csv('country_area.csv')

data = [go.Choropleth(locations=country_area['Code'],
                      z=country_area['Area'],
                      text=country_area['Country'],
                      colorscale=[[0, "rgb(5, 10, 172)"],
                                  [0.8, "rgb(44, 202, 238)"],
                                  [1, "rgb(220, 220, 220)"]],
                      autocolorscale=False,
                      reversescale=True)
        ]

layout = dict(
    title='Country Area (sqkm)',
)

fig = go.Figure(data=data, layout=layout)
plot(fig, filename='CountryArea.html')
