import jraph
import metadata
import pandas as pd
import plotnine as gg
import networkx as nx

def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
  nodes, edges, receivers, senders, _, _, _ = jraph_graph
  nx_graph = nx.DiGraph()
  if nodes is None:
    for n in range(jraph_graph.n_node[0]):
      nx_graph.add_node(n)
  else:
    for n in range(jraph_graph.n_node[0]):
      nx_graph.add_node(n, node_feature=nodes[n])
  if edges is None:
    for e in range(jraph_graph.n_edge[0]):
      nx_graph.add_edge(int(senders[e]), int(receivers[e]))
  else:
    for e in range(jraph_graph.n_edge[0]):
      nx_graph.add_edge(
          int(senders[e]), int(receivers[e]), edge_feature=edges[e])
  return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
  nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
  pos = nx.spring_layout(nx_graph)
  nx.draw(
      nx_graph, pos=pos, with_labels=True, node_size=500,
      labels=metadata.NODE_IDS_TO_LABELS_MAPPING
      )

def plot_samples(truth, prediction,
                 truth_label='truth', prediction_label='predicted'):
  assert truth.shape == prediction.shape
  df = pd.DataFrame({truth_label: truth.squeeze(), prediction_label: prediction.squeeze()}).reset_index()
  df = pd.melt(df, id_vars=['index'], value_vars=[truth_label, prediction_label])
  plot = (
      gg.ggplot(df)
      + gg.aes(x='index', y='value', color='variable')
      + gg.geom_line()
  )
  return plot