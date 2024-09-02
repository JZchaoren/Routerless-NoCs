import os
from tkinter import Y

def compute_layer(index_of_layer):
    num_rows = 2*index_of_layer
    rings = []

    # Compute A
    A = []
    A.append(compute_rectangle(0,num_rows-1,0,num_rows-1,'ccw'))

    # Compute B and C
    BC = []
    for i in range(num_rows-2):
        BC.append(compute_rectangle(0,i+1,0,num_rows-1,'cw'))
        BC.append(compute_rectangle(i+1,num_rows-1,0,num_rows-1,'cw'))

    # Compute D
    D = []
    if index_of_layer != 1:
        for i in range(num_rows-1):
            D.append(compute_rectangle(0,num_rows-1,i,i+1,'cw'))

    rings.extend(A)
    rings.extend(BC)
    rings.extend(D)

    return rings

def compute_rectangle(x_s, x_e, y_s, y_e, dipiction):
    output = [(x_s,y_s)]
    for i in range(y_s+1, y_e+1):
        output.append((x_s,i))
    for i in range(x_s+1, x_e+1):
        output.append((i,y_e))
    for i in range(y_e-1, y_s-1, -1):
        output.append((x_e,i))
    for i in range(x_e-1, x_s-1, -1):
        output.append((i, y_s))

    if dipiction == "ccw":
        return output
    elif dipiction == 'cw':
        output.reverse()
        return output
    else:
        raise ValueError("Invalid depiction: must be 'ccw' or 'cw'")

def compose_layers(N):
    solution = []
    for i in range(int(N/2)):
        a = int(N/2) - i - 1
        updated_nodes = add_to_coordinates(compute_layer(i+1), a)
        solution.extend(updated_nodes)
    return solution

def add_to_coordinates(main_list, a):
    for i in range(len(main_list)):
        for j in range(len(main_list[i])):
            x, y = main_list[i][j]
            main_list[i][j] = (x + a, y + a)
    return main_list

def compute_distance(N, k, ring_vertices):
    distance = 0
    for xa in range(N):
        for ya in range(N):
            for xb in range(N):
                for yb in range(N):
                    # 对于每一对A,B点来说：
                    # 如果两个点重合了，就跳过
                    if xa == xb and ya == yb:
                        continue
                    distance_ab = []
                    # 计算每个环上A,B的距离
                    for index in range(k):
                        ring = ring_vertices[index]
                        distance_ab.append(compute_distance_of_pair(xa, ya, xb, yb, ring))
                    # 选出最小的距离，作为distance_ab的最终距离
                    distance += min(distance_ab)

    distance /= (N ** 2) * (N ** 2 - 1)
    return distance

def compute_distance_of_pair(xa, ya, xb, yb, ring):
    point_a = (xa, ya)
    point_b = (xb, yb)

    if point_a not in ring or point_b not in ring:
        return 100000

    index_a = ring.index(point_a)
    index_b = ring.index(point_b)

    if index_a <= index_b:
        distance_ab = index_b - index_a
    else:
        distance_ab = len(ring) - (index_a - index_b) - 1

    return distance_ab

def count_edges_in_rings(rings):
    edge_count = {}

    for ring in rings:
        for i in range(len(ring)-1):
            start = ring[i]
            end = ring[i+1]

            edge = (start, end)  # Sequential

            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    return edge_count

def construct_routing_table(rings, N, output_file='./Experiment Data/Routerless/routing_table.txt'):
    output = {}

    for x1 in range(N):
        for y1 in range(N):
            for x2 in range(N):
                for y2 in range(N):
                    if (x1, y1) == (x2, y2):
                        continue

                    pair_key = ((x1, y1), (x2, y2))
                    output[pair_key] = {}

                    for ring_id, ring in enumerate(rings):
                        if (x1, y1) in ring and (x2, y2) in ring:
                            # distance
                            idx1 = ring.index((x1, y1))
                            idx2 = ring.index((x2, y2))

                            if idx1 <= idx2:
                                distance_ab = idx2 - idx1
                            else:
                                distance_ab = len(ring) - (idx1 - idx2) - 1

                            output[pair_key][ring_id] = distance_ab

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for pair_key, rings_dict in output.items():
            a, b = pair_key
            xa, ya = a
            xb, yb = b
            nodea = xa + N * ya
            nodeb = xb + N * yb
            f.write(f"Pair {(nodea, nodeb)}:\n")
            for ring_id, length in sorted(rings_dict.items(), key=lambda item: item[1]):
                f.write(f"  Ring {ring_id}: Length = {length}\n")
            f.write("\n")

    return output

def evaluate(N, k, ring_vertices, lam):
    lenth = 0
    for index in range(k):
        lenth += len(ring_vertices[index]) - 1  # Lenth of each ring

    distance = compute_distance(N, k, ring_vertices)

    score = distance + lam * lenth
    return score, distance, lenth

if __name__ == "__main__":
    N = 4
    ring_vertices = compose_layers(N)
    print(evaluate(N, 9, ring_vertices, 0.01))
    construct_routing_table(ring_vertices, N, output_file='./Experiment Data/Routerless/routing_table.txt')
