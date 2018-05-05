import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('dark_background')

year = [i for i in range(2012, 2017 + 1)]
revenue = [1118000, 1156000, 1233000, 1280000, 1201000, 1232000]

# dollarFormat = '${x:,.0f}'
#
# fig, ax = plt.subplots(1, 1)
# tick = mtick.StrMethodFormatter(dollarFormat)
# ax.yaxis.set_major_formatter(tick)
# ax.plot(year, revenue)
# ax.set_ylim([1000000, 1500000])
# ax.set_xlabel('year')
# ax.set_title('Company Revenue (2012 - 2017)')
# plt.show()


import plotly.offline as py
import plotly.graph_objs as go

data = [
    go.Scatter(x=year, y=revenue)
]
layout = go.Layout(
    autosize=True,
    yaxis=dict(tickformat="$,.2s")
)


fig = go.Figure(data=data, layout=layout)
py.plot(fig)

