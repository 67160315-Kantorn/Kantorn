import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class graph_structure:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neighbor):
        """เพิ่มเส้นเชื่อมระหว่าง node และ neighbor"""
        if node not in self.graph:
            self.graph[node] = []
        if neighbor not in self.graph:
            self.graph[neighbor] = []
        self.graph[node].append(neighbor)
        self.graph[neighbor].append(node)  # ถ้าอยากให้เป็น directed ให้ลบบรรทัดนี้ออก

    def show_graph(self):
        """แสดงข้อมูลโครงสร้างกราฟ"""
        print("=== Graph Structure ===")
        for node, neighbors in self.graph.items():
            print(f"{node} -> {neighbors}")

    def plot_graph(self, highlight_nodes=None, title="Graph Structure"):
        """วาดกราฟด้วย NetworkX"""
        G = nx.Graph(self.graph)
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 4))

        node_colors = []
        for n in G.nodes():
            if highlight_nodes and n in highlight_nodes:
                node_colors.append('lightcoral')
            else:
                node_colors.append('skyblue')

        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1200,
            font_size=12,
            font_weight='bold',
            edge_color='gray'
        )
        plt.title(title)
        plt.show()

    def bfs(self, start):
        """Breadth-First Search"""
        visited = set()
        queue = deque([start])
        order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                order.append(node)
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
        print("BFS Order:", order)
        self.plot_graph(highlight_nodes=order, title="Breadth-First Search")
        return order

    def dfs(self, start):
        """Depth-First Search"""
        visited = set()
        order = []

        def dfs_recursive(node):
            visited.add(node)
            order.append(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    dfs_recursive(neighbor)

        dfs_recursive(start)
        print("DFS Order:", order)
        self.plot_graph(highlight_nodes=order, title="Depth-First Search")
        return order


# ====== ตัวอย่างการใช้งาน ======
if __name__ == "__main__":
    g = graph_structure()
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'D')
    g.add_edge('C', 'E')
    g.add_edge('D', 'F')
    g.add_edge('E', 'F')

    g.show_graph()
    g.bfs('A')
    g.dfs('A')
