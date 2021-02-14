import plotly.graph_objects as go
import pandas as pd
import chart_studio
import chart_studio.plotly as py

# list1 = [[.4,.6,.7,.567,.643,.754],[.3,.23,.5,.234,.543,.435],[.4,.7,.2,.9,.65,.3456],[.4,.6,.7,.567,.643,.754],[.3,.23,.5,.234,.543,.435],[.4,.7,.2,.9,.65,.3456]]
# text_units = ['book','read','wall','science','semantic','analysis']

def landscape_surfaceplot(list_of_lists_recal_probabilities, list_textUnits):
  username = 'sibhat'
  api_key = 'ywPLaxlgYgEghyysH6dx'
  chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
  cycles = [i for i in range(1,len(list_of_lists_recal_probabilities[0])+1)]
  z_data = pd.DataFrame(list_of_lists_recal_probabilities,
                        index = list_textUnits,columns=cycles)
  fig = go.Figure(data=[go.Surface(z=z_data.values, x=z_data.columns,y=z_data.index)])
  fig.update_layout(title='Landscape Model', autosize=False,
                    width=750, height=750)
  b = py.plot(fig, filename='landscape_model', auto_open=True)
  return b

# a = landscape_surfaceplot(list1,text_units)
