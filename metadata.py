NUM_NODES = 6
NUM_EDGES = 15

ADJACENCY_MATRIX = [
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
]

CONNECTIONS = [
    # B1
    ("b1", "ftj"),
    ("b1", "uci"),
    ("b1", "b6"),
    # B6
    ("b6", "ms"),
    ("b6", "uci"),
    ("b6", "b1"),
    ("b6", "ftj"),
    # UCI
    ("uci", "ftj"),
    ("uci", "b1"),
    ("uci", "b6"),
    ("uci", "cyfronet"),
     # FTJ
    ("ftj", "b1"),
    ("ftj", "uci"),
    ("ftj", "b6"),
    ("ftj", "cyfronet"),
    ]

NODE_IDS_TO_LABELS_MAPPING = {
    0: "cyfronet",
    1: "uci",
    2: "ftj",
    3: "b1",
    4: "b6",
    5: "ms"
}